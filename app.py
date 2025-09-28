import json
import os
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from llama_cpp import Llama

app = FastAPI(title="llama-cpp FastAPI (RunPod LB)")

# ── Load model once at startup, using your env-driven config ────────────────
# Reads LLAMA_ARGS (JSON). Example:
#   export LLAMA_ARGS='{"model":"/models/DeepSeek-R1-8B.gguf","n_ctx":32768,"n_gpu_layers":-1,"flash_attn":true}'
RAW = os.environ.get("LLAMA_ARGS", "{}")
try:
    ARGS = json.loads(RAW)
except json.JSONDecodeError as e:
    raise RuntimeError(f"LLAMA_ARGS must be valid JSON: {e}")

# Safe defaults (kept from your handle.py, with minor fix for 'model')
ARGS.setdefault("n_ctx", 32768)
ARGS.setdefault("n_gpu_layers", -1)   # offload all layers if VRAM allows
ARGS.setdefault("flash_attn", True)
ARGS.setdefault("model_path", "DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf")

# Create global model
llm = Llama(**ARGS)


# ── Request/response models (simple, OpenAI-ish) ────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stop: Optional[List[str]] = None

# If you want to return only the string, change this to a simpler schema.
# For now we return the raw llama-cpp dict for familiarity.
CompletionDict = Dict[str, Any]


# ── Utilities ───────────────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "4096"))
MAX_TOKENS_CAP     = int(os.environ.get("MAX_TOKENS_CAP", "4096"))

def _inject_defaults(payload: ChatRequest) -> Dict[str, Any]:
    """
    Convert ChatRequest to kwargs for llama.create_chat_completion with defaults/caps.
    """
    kw: Dict[str, Any] = payload.model_dump(exclude_none=True)
    # Ensure a default and clamp hard
    requested = kw.get("max_tokens", DEFAULT_MAX_TOKENS)
    kw["max_tokens"] = min(int(requested), MAX_TOKENS_CAP)
    return kw

async def _chat_call(kwargs: Dict[str, Any]) -> CompletionDict:
    # llama-cpp call is blocking; run it in a thread
    return await asyncio.to_thread(llm.create_chat_completion, **kwargs)


@app.get("/ping")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": os.path.basename(ARGS.get("model", ""))}

@app.post("/generate")
async def chat(req: ChatRequest) -> CompletionDict:
    try:
        kwargs = _inject_defaults(req)
        # llama-cpp expects messages as list[{"role","content"}] — already aligned
        out = await _chat_call(kwargs)
        return out
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Surface a concise error to clients
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")

if __name__ == "__main__":
    import uvicorn

    # When you deploy the endpoint, make sure to expose port 5000
    # And add it as an environment variable in the Runpod console
    port = int(os.getenv("PORT", "8080"))

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)
