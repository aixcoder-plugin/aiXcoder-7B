# Code adapted from https://github.com/bigcode-project/starcoder2/blob/main/finetune.py
import argparse
import multiprocessing
import os
import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging,
    set_seed,
)
import numpy as np
import random
import warnings
import sys
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="aiXcoder/aixcoder-7b-base")
    parser.add_argument("--dataset_name", type=str, default="the-stack-smol")
    parser.add_argument("--subset", type=str, default="data/rust")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--fim_rate", type=float, default=0.5)
    parser.add_argument("--dataset_text_field", type=str, default="content")

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_aix_7b")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    return parser.parse_args()


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True, file=sys.stderr)
    else:
        print(message, flush=True)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print_rank_0(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class RandomFIMDataset(ConstantLengthDataset):
    """
        This class supports the random fill-in-the-middle (FIM) task. If `fim_rate` is greater than 0, 
        it constructs data in the fill-in-the-middle format with a probability of `fim_rate`. 
        The aiXcoder-7b-base model uses structured FIM during pre-training, 
        where a complete code block is constructed as the MIDDLE. 
        However, creating such training data involves syntactic parsing, 
        and we currently do not plan to open source the processing code.
    
    """
    def __init__(self, tokenizer, dataset, dataset_text_field=None, fim_rate=0, formatting_func=None, infinite=False, seq_length=1024, num_of_sequences=1024, chars_per_token=3.6, eos_token_id=0, shuffle=True, append_concat_token=True, add_special_tokens=True):
        self.fim_rate = fim_rate
        self.fim_spm_rate = 0.5
        self.np_rand = np.random.RandomState(seed=3574)
        if self.fim_rate > 0:
            print_rank_0(f"constructing data wit FIM: fim_rate: {self.fim_rate}")
        super().__init__(tokenizer, dataset, dataset_text_field, formatting_func, infinite, seq_length, num_of_sequences, chars_per_token, eos_token_id, shuffle, append_concat_token, add_special_tokens)
    
    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    if self.fim_rate > 0:
                        if self.np_rand.binomial(1, self.fim_rate): # sample bernoulli dist
 
                            contents = self.formatting_func(next(iterator))
                            
                            try:
                                boundaries = list(self.np_rand.randint(low=0, high=len(contents) + 1, size=2))
                                boundaries.sort()
                            except ValueError as e:
                                print(len(contents), contents)
                                print(e)
                                raise e

                            prefix = contents[:boundaries[0]]
                            middle = contents[boundaries[0]:boundaries[1]]
                            suffix = contents[boundaries[1]:]
                            if self.np_rand.binomial(1, self.fim_spm_rate):
                                contents = f"<s>▁<AIX-SPAN-PRE>▁<AIX-SPAN-POST>{suffix}▁<AIX-SPAN-MIDDLE>{prefix}{middle}</s>"
                            else:
                                contents = f"<s>▁<AIX-SPAN-PRE>{prefix}▁<AIX-SPAN-POST>{suffix}▁<AIX-SPAN-MIDDLE>{middle}</s>"
                        else:
                            contents = f"<s>{self.formatting_func(next(iterator))}</s>"
                    else:
                        contents = f"<s>{self.formatting_func(next(iterator))}</s>"
                            
                    buffer.append(contents)
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, add_special_tokens=self.add_special_tokens, truncation=False)[
                "input_ids"
            ]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


def main(args):
    # config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # load model and dataset
    token = os.environ.get("HF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=args.attention_dropout,
        attn_implementation='flash_attention_2'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print_trainable_parameters(model)

    data = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        token=token,
        num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
    )

    train_data = RandomFIMDataset(
        tokenizer=tokenizer, dataset=data, fim_rate=args.fim_rate, dataset_text_field=args.dataset_text_field,
        infinite=True, seq_length=args.max_seq_length, eos_token_id=tokenizer.eos_token_id
    )

    # setup the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        max_seq_length=args.max_seq_length,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            logging_strategy="steps",
            logging_steps=10,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
            report_to="none",
        ),
        peft_config=lora_config,
        dataset_text_field=args.dataset_text_field,
    )

    # launch
    print_rank_0("Training...")
    trainer.train()

    print_rank_0("Saving the last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    if args.push_to_hub:
        trainer.push_to_hub("Upload model")
    print_rank_0("Training Done! ")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)