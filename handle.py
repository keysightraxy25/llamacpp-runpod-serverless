import json
import os
import glob

import runpod
from llama_cpp import Llama

args = json.loads(os.environ.get("LLAMA_ARGS", "{}"))
args.setdefault("n_ctx", 32768)
args.setdefault("n_gpu_layers", -1)
args.setdefault("flash_attn", True)
args.setdefault("model_path", "/models/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf")
llm = Llama(**args)

def handler(event):
    """
    This function is the entry point for the serverless handler.
    """
    inp = event.get("input", {}) or {}

    if "list_models" in inp:
        return str(glob.glob(os.path.join("/models", "*.gguf")))

    inp.setdefault("max_tokens", 4096)

    return llm.create_chat_completion(**inp)

runpod.serverless.start({"handler": handler})
