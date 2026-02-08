#!/usr/bin/env python3
"""Interactive chat with a local Llama model using llama.cpp."""

import json
import os
from pathlib import Path

import click
from llama_cpp import Llama
from rich.console import Console
from rich.markdown import Markdown

console = Console()
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_configs.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def find_model_path(model: str, quant: str, model_path: str, config: dict) -> str:
    """Resolve the path to the GGUF model file."""
    # Direct path takes priority
    if model_path:
        if os.path.exists(model_path):
            return model_path
        console.print(f"[red]Model file not found: {model_path}[/red]")
        raise SystemExit(1)

    # Look up from config
    model_config = config["models"][model]
    if quant is None:
        quant = model_config["default_quant"]

    filename = model_config["filenames"].get(quant)
    if not filename:
        console.print(f"[red]Unknown quantization: {quant}[/red]")
        raise SystemExit(1)

    models_dir = config.get("models_dir", "./models")
    path = os.path.join(models_dir, filename)

    if not os.path.exists(path):
        console.print(f"[red]Model not found at: {path}[/red]")
        console.print(f"Download it first:")
        console.print(f"  [cyan]python scripts/download_model.py --model {model} --quant {quant}[/cyan]")
        raise SystemExit(1)

    return path


def format_messages(messages: list[dict]) -> str:
    """Format messages into Llama 3.1 chat template."""
    formatted = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted


@click.command()
@click.option("--model", type=click.Choice(["8b", "70b"]), default="8b",
              help="Model size")
@click.option("--quant", type=str, default=None,
              help="Quantization level")
@click.option("--model-path", type=str, default=None,
              help="Direct path to a GGUF file (overrides --model)")
@click.option("--system", type=str, default="You are a helpful assistant.",
              help="System prompt")
@click.option("--ctx", type=int, default=4096,
              help="Context window size")
@click.option("--gpu-layers", type=int, default=-1,
              help="Layers to offload to GPU (-1 = all, 0 = CPU only)")
@click.option("--temp", type=float, default=0.7,
              help="Temperature for generation")
def chat(model, quant, model_path, system, ctx, gpu_layers, temp):
    """Interactive chat with a local Llama model."""
    config = load_config()
    path = find_model_path(model, quant, model_path, config)

    console.print(f"\n[bold blue]Loading model: {os.path.basename(path)}[/bold blue]")
    console.print(f"  Context: {ctx} tokens | GPU layers: {gpu_layers} | Temp: {temp}")
    console.print()

    llm = Llama(
        model_path=path,
        n_ctx=ctx,
        n_gpu_layers=gpu_layers,
        verbose=False,
    )

    console.print("[bold green]Model loaded. Type 'quit' to exit, 'clear' to reset.[/bold green]\n")

    messages = [{"role": "system", "content": system}]

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": system}]
            console.print("[dim]Conversation cleared.[/dim]\n")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = format_messages(messages)

        console.print("[bold green]Assistant:[/bold green] ", end="")

        response_text = ""
        for chunk in llm(
            prompt,
            max_tokens=1024,
            temperature=temp,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            response_text += token
            console.print(token, end="", highlight=False)

        console.print("\n")
        messages.append({"role": "assistant", "content": response_text.strip()})


if __name__ == "__main__":
    chat()
