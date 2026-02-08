#!/usr/bin/env python3
"""Download quantized Llama GGUF models from Hugging Face."""

import json
import os
from pathlib import Path

import click
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.progress import Progress

console = Console()
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_configs.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


@click.command()
@click.option("--model", type=click.Choice(["8b", "70b"]), default="8b",
              help="Model size to download")
@click.option("--quant", type=str, default=None,
              help="Quantization level (e.g., Q4_K_M, Q8_0). Defaults to model's default.")
@click.option("--output-dir", type=str, default=None,
              help="Directory to save model. Defaults to ./models/")
def download(model: str, quant: str, output_dir: str):
    """Download a quantized Llama model."""
    config = load_config()
    model_config = config["models"][model]

    if quant is None:
        quant = model_config["default_quant"]

    if quant not in model_config["filenames"]:
        available = ", ".join(model_config["filenames"].keys())
        console.print(f"[red]Quantization '{quant}' not available. Options: {available}[/red]")
        return

    filename = model_config["filenames"][quant]
    repo_id = model_config["repo_id"]

    if output_dir is None:
        output_dir = config.get("models_dir", "./models")

    os.makedirs(output_dir, exist_ok=True)

    console.print(f"\n[bold blue]Downloading Llama 3.1 {model.upper()}[/bold blue]")
    console.print(f"  Repo:     {repo_id}")
    console.print(f"  File:     {filename}")
    console.print(f"  Quant:    {quant}")
    console.print(f"  Save to:  {output_dir}/")
    console.print()

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        console.print(f"\n[bold green]✓ Downloaded to: {path}[/bold green]")
        console.print(f"\nRun it with:")
        console.print(f"  [cyan]python scripts/chat.py --model {model} --quant {quant}[/cyan]")

    except Exception as e:
        console.print(f"\n[red]Download failed: {e}[/red]")
        console.print("\nMake sure you've:")
        console.print("  1. Accepted the Llama license on Hugging Face")
        console.print("  2. Run: huggingface-cli login")


if __name__ == "__main__":
    download()
