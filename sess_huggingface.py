import torch
import sys
from hf_mini.utils import input_wrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

tokenizer = AutoTokenizer.from_pretrained("aiXcoder/aixcoder-7b-base")
model = AutoModelForCausalLM.from_pretrained("aiXcoder/aixcoder-7b-base", torch_dtype=torch.bfloat16)


text = input_wrapper(
    code_string="# 快速排序算法",
    later_code="\n",
    path="test.py"
)

if len(text) == 0:
    sys.exit()

inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False)

inputs = inputs.to(device)
model.to(device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
