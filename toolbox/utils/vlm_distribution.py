from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import sys
sys.path.append('/home/zhaosiyao/SoccerAgent/toolbox')

# Instead of using a single hardcoded device, we will load the model across available GPUs automatically.
# We will exclude cuda:4 if you want to reserve it for unisoccer, or just let accelerate handle it automatically.
# To avoid OOM during large context reasoning, device_map="auto" is the best practice.

VLM_MODEL_PATH = "/data/zhaosiyao/model/Qwen-Qwen2.5-VL-7B-Instruct"
print("Loading VLM on multi-GPU with device_map='auto'...")

# Load the model across multiple GPUs automatically.
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
vlm_model.eval()

# default processer
vlm_processor = AutoProcessor.from_pretrained(
    VLM_MODEL_PATH,
    use_fast=True
)
