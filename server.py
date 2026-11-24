"""Student-facing API: guardrail, then cache-friendly chat messages, streamed from vLLM.

Start vLLM first:
    vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 16384 --enable-prefix-caching
Then:
    uvicorn server:app --port 8001
"""
import json
import os

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

import guardrail
from prompting import PromptBuilder

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

app = FastAPI(title="2am Office Hours")
builder = PromptBuilder(os.environ.get("INDEX_DIR", "index"))


class Ask(BaseModel):
    question: str
    max_tokens: int = 256


@app.post("/ask")
async def ask(req: Ask):
    refusal = guardrail.check(req.question)
    if refusal:
        return PlainTextResponse(refusal)

    payload = {"model": MODEL, "messages": builder.messages(req.question),
               "max_tokens": req.max_tokens, "temperature": 0.2, "stream": True}

    async def gen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", VLLM_URL, json=payload) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta

    return StreamingResponse(gen(), media_type="text/plain")


@app.get("/health")
def health():
    return {"ok": True, "prefix_chars": len(builder.prefix), "note_chunks": len(builder.chunks)}
