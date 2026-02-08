#!/usr/bin/env python3
"""REST API server for local Llama inference (OpenAI-compatible)."""

import json
import os
import time
from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_configs.json"

app = FastAPI(title="Llama Local API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global model reference
llm = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    choices: list[dict]
    usage: dict


def format_messages(messages: list[Message]) -> str:
    formatted = "<|begin_of_text|>"
    for msg in messages:
        formatted += f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}<|eot_id|>"
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": llm is not None}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = format_messages(request.messages)

    output = llm(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    response_text = output["choices"][0]["text"]

    return ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": response_text.strip()},
            "finish_reason": "stop",
        }],
        usage={
            "prompt_tokens": output["usage"]["prompt_tokens"],
            "completion_tokens": output["usage"]["completion_tokens"],
            "total_tokens": output["usage"]["total_tokens"],
        },
    )


@click.command()
@click.option("--model", type=click.Choice(["8b", "70b"]), default="8b")
@click.option("--quant", type=str, default=None)
@click.option("--model-path", type=str, default=None)
@click.option("--port", type=int, default=8000)
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--ctx", type=int, default=4096)
@click.option("--gpu-layers", type=int, default=-1)
def serve(model, quant, model_path, port, host, ctx, gpu_layers):
    """Start the Llama API server."""
    global llm

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    if model_path and os.path.exists(model_path):
        path = model_path
    else:
        model_config = config["models"][model]
        q = quant or model_config["default_quant"]
        filename = model_config["filenames"][q]
        path = os.path.join(config.get("models_dir", "./models"), filename)

    if not os.path.exists(path):
        print(f"Model not found: {path}")
        print(f"Run: python scripts/download_model.py --model {model}")
        return

    print(f"Loading model: {path}")
    llm = Llama(model_path=path, n_ctx=ctx, n_gpu_layers=gpu_layers, verbose=False)
    print(f"Model loaded. Starting server on {host}:{port}")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
