#!/usr/bin/env python3
"""
Convert a fine-tuned Llama model to GGUF format for local inference.

This script:
  1. Merges LoRA weights back into the base model
  2. Converts to GGUF using llama.cpp's convert script
  3. Quantizes to your chosen format (Q4_K_M, Q8_0, etc.)

Prerequisites:
  - Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp
  - Build it: cd llama.cpp && make
  - Or install llama-cpp-python (which bundles the converter)

Usage:
    python scripts/quantize.py \
        --model-path ./my-fine-tuned-model \
        --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --output ./models/my-model.gguf \
        --quant Q4_K_M
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--model-path", type=str, required=True,
              help="Path to the fine-tuned model (LoRA adapter or merged)")
@click.option("--base-model", type=str, default=None,
              help="Base model ID for LoRA merge (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
@click.option("--output", type=str, required=True,
              help="Output path for the GGUF file")
@click.option("--quant", type=str, default="Q4_K_M",
              help="Quantization type: Q4_K_M, Q5_K_M, Q8_0, F16")
@click.option("--llama-cpp-path", type=str, default=None,
              help="Path to llama.cpp directory (auto-detected if not set)")
def quantize(model_path, base_model, output, quant, llama_cpp_path):
    """Convert a fine-tuned model to quantized GGUF format."""

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    merged_path = model_path + "-merged"

    # ── Step 1: Merge LoRA if needed ──
    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        console.print("\n[bold blue]Step 1: Merging LoRA weights...[/bold blue]")

        if base_model is None:
            console.print("[red]--base-model is required when merging LoRA adapters[/red]")
            return

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            console.print(f"  Loading base model: {base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.float16, device_map="cpu"
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model)

            console.print(f"  Loading LoRA adapter: {model_path}")
            model = PeftModel.from_pretrained(model, model_path)

            console.print("  Merging weights...")
            model = model.merge_and_unload()

            console.print(f"  Saving merged model to: {merged_path}")
            model.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)

            model_path = merged_path
            console.print("[green]  ✓ LoRA weights merged[/green]")

        except ImportError:
            console.print("[red]Install training dependencies: pip install -r requirements-training.txt[/red]")
            return
    else:
        console.print("\n[dim]Step 1: No LoRA adapter found, assuming already merged.[/dim]")

    # ── Step 2: Convert to GGUF ──
    console.print("\n[bold blue]Step 2: Converting to GGUF...[/bold blue]")

    # Find llama.cpp convert script
    convert_script = None
    search_paths = [
        llama_cpp_path,
        os.environ.get("LLAMA_CPP_PATH"),
        os.path.expanduser("~/llama.cpp"),
        "./llama.cpp",
    ]

    for p in search_paths:
        if p and os.path.exists(os.path.join(p, "convert_hf_to_gguf.py")):
            convert_script = os.path.join(p, "convert_hf_to_gguf.py")
            break

    if convert_script is None:
        console.print("[yellow]llama.cpp not found. Attempting pip-installed converter...[/yellow]")
        # Try using the python package
        try:
            f16_output = output.replace(".gguf", "-f16.gguf")
            subprocess.run([
                sys.executable, "-m", "llama_cpp.convert",
                model_path, "--outfile", f16_output, "--outtype", "f16"
            ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("\n[red]Could not find a GGUF converter.[/red]")
            console.print("Please either:")
            console.print("  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
            console.print("  2. Set --llama-cpp-path to your llama.cpp directory")
            return
    else:
        f16_output = output.replace(".gguf", "-f16.gguf")
        console.print(f"  Using converter: {convert_script}")
        subprocess.run([
            sys.executable, convert_script,
            model_path, "--outfile", f16_output, "--outtype", "f16"
        ], check=True)

    console.print(f"[green]  ✓ F16 GGUF created: {f16_output}[/green]")

    # ── Step 3: Quantize ──
    if quant != "F16":
        console.print(f"\n[bold blue]Step 3: Quantizing to {quant}...[/bold blue]")

        # Find llama-quantize binary
        quantize_bin = None
        for p in search_paths:
            if p:
                for name in ["llama-quantize", "quantize"]:
                    candidate = os.path.join(p, name)
                    if os.path.exists(candidate):
                        quantize_bin = candidate
                        break

        if quantize_bin is None:
            console.print("[yellow]llama-quantize not found. Build llama.cpp first:[/yellow]")
            console.print("  cd llama.cpp && make llama-quantize")
            console.print(f"\nThen run manually:")
            console.print(f"  ./llama.cpp/llama-quantize {f16_output} {output} {quant}")
            return

        subprocess.run([quantize_bin, f16_output, output, quant], check=True)

        # Clean up F16 intermediate
        if os.path.exists(output):
            os.remove(f16_output)
            console.print(f"[green]  ✓ Quantized to {quant}: {output}[/green]")
    else:
        shutil.move(f16_output, output)

    # Clean up merged model
    if os.path.exists(merged_path):
        console.print(f"\n[dim]Cleaning up merged model at {merged_path}...[/dim]")
        shutil.rmtree(merged_path)

    console.print(f"\n[bold green]Done! Your model is ready at: {output}[/bold green]")
    console.print(f"\nRun it:")
    console.print(f"  [cyan]python scripts/chat.py --model-path {output}[/cyan]")


if __name__ == "__main__":
    quantize()
