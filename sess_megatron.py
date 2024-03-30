import sys
import os
import pathlib
import torch
import traceback
import numpy as np
from typing import List, Tuple
from megatron_mini import get_args
from megatron_mini.initialize import initialize_megatron
from megatron_mini.model import LLaMAModel
from megatron_mini.utils import get_model_for_infer, Tokenizer


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True, file=sys.stderr)
    else:
        print(message, flush=True, file=sys.stderr)


def add_code_generation_args(parser):
    """Code generation arguments."""
    group = parser.add_argument_group(title="code generation")

    group.add_argument(
        "--padded_vocab_size",
        type=int,
        default=40000,
        help="Start id for whitespace encoding",
    )
    group.add_argument("--model_dir", type=str, default="")
    group.add_argument("--model_name", type=str, default="aix3-7b")

    return parser

class Predictor(object):

    def __init__(self, args):
        
        self.args = args
        self.checkpoint_head_hash: str = ""
        self.np_rand = np.random.RandomState(seed=1414)

        # build predictor
        self.tokenizer = self.create_tokenizer()
        
        self.dtype = torch.float32
        if self.args.bf16:
            self.dtype = torch.bfloat16
        elif self.args.fp16:
            self.dtype = torch.half
        
        self.predictor = self.create_predictor()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    @staticmethod
    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        
        print_rank_0("Building Codemodel ...")
        model = LLaMAModel(parallel_output=False)
        return model
    
    @staticmethod
    def pad_batch(tokens_id, max_seq_len=2048):
        """
            pad_batch was used by syncing token_ids
        """
        tokens_id = np.reshape(tokens_id, [1, -1])
        context_length = tokens_id.shape[-1]

        assert context_length <= max_seq_len, f"{context_length}, {max_seq_len}"

        if context_length < max_seq_len:
            tokens_id = np.concatenate([tokens_id, np.zeros(shape=[1, max_seq_len-context_length], dtype=tokens_id.dtype)], axis=-1)
        return tokens_id.astype(np.int64), np.array([context_length], dtype=np.int64)

    
    @staticmethod
    def sync_type_info(sess_id: int) -> int:
        input_info = np.array([sess_id], dtype=np.int64)
        input_info_tensor = torch.tensor(input_info, dtype=torch.int64, device='cuda')
        torch.distributed.broadcast(
                input_info_tensor,
                0,
            )
        sess_id = input_info_tensor[0].item()
        return sess_id

    @staticmethod
    def sync_obj_info(model_dir: str) -> str:

        tmp_list = [model_dir]
        torch.distributed.broadcast_object_list(
                tmp_list,
                0,
            )
        return tmp_list[0]

    def create_predictor(self):
        
        model_dir = self.args.model_dir
        
        assert self.args.num_attention_heads % self.args.tensor_model_parallel_size == 0
        assert self.args.hidden_size % self.args.num_attention_heads == 0
        
        model = get_model_for_infer(self.model_provider)
        print_rank_0("Loading state dict ...")
        _ = self.load_checkpoint(model, model_dir)
        
        assert len(model) == 1, "Above condition should have caught this"
        
        model = model[0]
        model.eval()
        
        if self.args.bf16 or self.args.fp16 :
            print_rank_0(f" > converting model to {'bf16' if self.args.bf16 else 'fp16'} ...")
            model.to(self.dtype)
        
        print_rank_0(f" > moving model to GPU ...")
        model.cuda(torch.cuda.current_device())
        
        return model
    
    def create_tokenizer(self):
        assert os.path.exists(os.path.join(self.args.model_dir, "tokenizer.model"))
        tokenizer = Tokenizer(model_path=os.path.join(self.args.model_dir, "tokenizer.model"))
        return tokenizer

    def load_checkpoint(self, model: List[LLaMAModel], path):
        
        assert isinstance(model, list)

        if not (path is not None and os.path.exists(path)):
            raise ValueError

        iteration = 0
        if self.args.tensor_model_parallel_size == 1 and self.args.rank < self.args.tensor_model_parallel_size:
            checkpoint_name = os.path.join(path, f"{self.args.model_name}.pt")
            assert os.path.isfile(checkpoint_name)
        elif self.args.rank < self.args.tensor_model_parallel_size:
            checkpoints = sorted(pathlib.Path(path).glob(f"{self.args.model_name}_states_*.pt"))
            assert len(checkpoints) == self.args.tensor_model_parallel_size
            checkpoint_name = checkpoints[self.args.rank]
        else:
            raise ValueError

        # Load the checkpoint.
        print(f"rank_{self.args.rank} load: {checkpoint_name}", flush=True, file=sys.stderr)
        state_dict = torch.load(checkpoint_name, map_location="cpu")

        # Set iteration.
        iteration = state_dict.get("iteration", 0)

        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "module" in state_dict:
            state_dict = state_dict["module"]

        # Model.
        model[0].load_state_dict(state_dict, strict=True)

        print_rank_0(
            f"successfully loaded checkpoint from {path} "
            f"at iteration {iteration}"
        )

        return iteration
    
    def predict_batch(self, data):
        
        common_len = int(data[1].item())

        with torch.no_grad():
            tokens_ids = data[0].clone().detach().cuda()
            
            logits = self.predictor(
                tokens=tokens_ids,         # shape: [bsz, 1024]
                start_pos=common_len,
            )
            logits = logits[:, -1].view(1, -1).contiguous()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            return [np.squeeze(probs)]

    def predict(self, token_ids: List[int], common_len: int) -> Tuple[List[int], List[float]]:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        try:
            common_len_nda = np.array([common_len]).astype("int64")
            token_ids_nda = np.array([token_ids], dtype=np.int64)

            max_pad_len = max(token_ids_nda.shape[-1], 128)
            max_pad_len = self.sync_type_info(max_pad_len)
            token_ids_nda, tokens_id_len = self.pad_batch(token_ids_nda, max_seq_len=max_pad_len)

            context_tensor = torch.tensor(token_ids_nda, dtype=torch.int64, device='cuda')
            context_tensor_length = torch.tensor(tokens_id_len, dtype=torch.int64, device='cuda')
            context_common_len = torch.tensor(common_len_nda, dtype=torch.int64, device='cuda')

            torch.distributed.broadcast(
                context_tensor,
                0,
            )
            torch.distributed.broadcast(
                context_tensor_length,
                0,
            )
            torch.distributed.broadcast(
                context_common_len,
                0,
            )
            
            tokens_id_len = context_tensor_length.min().item()
            batch = [context_tensor[:, :tokens_id_len], context_common_len]
            
            out = self.predict_batch(batch)

            # shape: [bsz, vocab_size] => [vocab_size]
            out = out[0]
            
            predict_id = np.argmax(out)
            return [int(predict_id)], [out[predict_id]]
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(e)


class TestInference:
    def __init__(self) -> None:
        aix_config = {
            "num_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
            "max_position_embeddings": 32768, "fp16": False, "bf16": True, 
            "rope_theta": 256000, "inner_hidden_dim": 14464, "padded_vocab_size": 49152, 
            "seq_length": 4096, "micro_batch_size": 1, "use_flash_attn": True,
            "use_cpu_initialization": True, "attention_head_type": "groupedquery"
        }
        initialize_megatron(
            extra_args_provider=add_code_generation_args,
            aix_config=aix_config
        )
        args = get_args()

        self.sess = Predictor(args=args)
        self.end_token_set = self.sess.tokenizer.end_token_set
    
    def run_infer(self, code_string: str, max_new_tokens: int = 256, later_code: str = "", file_path: str = "") -> None:
        tokens = self.sess.tokenizer.encode(
            code_string=code_string, later_code=later_code, file_path=file_path
        )

        predict_list = []
        common_len = 0
        while True:

            if torch.distributed.get_rank() == 0:

                output_vals = self.sess.predict(
                    np.array([tokens], dtype='int32'),
                    np.array([common_len], dtype='int32')
                    )
                
                predict_list.append(output_vals[0][0])

                if len(predict_list) >= max_new_tokens or predict_list[-1] in self.end_token_set:
                    terminate_runs = 1
                else:
                    terminate_runs = 0
                
                common_len += len(tokens)
                tokens = predict_list[-1:]

            else:
                tokens = [0] * 4
                output_vals = self.sess.predict([], [], input_vals=[
                    np.array([tokens], dtype='int32'),
                    np.array([0], dtype='int32')
                    ])
                predict_list.append(0)
                terminate_runs = 0

            
            if self.sess.sync_type_info(terminate_runs) > 0:
                break
        return self.sess.sync_obj_info(self.sess.tokenizer.decode(predict_list))

if __name__ == "__main__":
    infer = TestInference()
    res = infer.run_infer(
        code_string="""# 快速排序算法""",
        later_code="\n",
        file_path="test.py"
    )

    print(res)
