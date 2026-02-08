#!/usr/bin/env python3
"""
Fine-tune Llama 3.1 with LoRA/QLoRA.

Run this on a cloud GPU (Colab T4, A100, etc.), not your laptop.

Usage:
    python scripts/fine_tune.py \
        --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --dataset data/train.jsonl \
        --output ./my-fine-tuned-model \
        --epochs 3
"""

import json
from pathlib import Path

import click
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "training_config.json"


def load_training_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def format_example(example: dict) -> str:
    """Format a training example into Llama 3.1 chat format."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    response = example.get("response", example.get("output", ""))

    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{response}<|eot_id|>"
        "<|end_of_text|>"
    )


@click.command()
@click.option("--base-model", type=str, required=True,
              help="HuggingFace model ID (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
@click.option("--dataset", type=str, required=True,
              help="Path to JSONL training data")
@click.option("--output", type=str, default="./fine-tuned-model",
              help="Output directory for the fine-tuned model")
@click.option("--epochs", type=int, default=3,
              help="Number of training epochs")
@click.option("--batch-size", type=int, default=4,
              help="Per-device batch size")
@click.option("--lr", type=float, default=2e-4,
              help="Learning rate")
@click.option("--max-seq-length", type=int, default=2048,
              help="Maximum sequence length")
@click.option("--no-quantize", is_flag=True, default=False,
              help="Disable 4-bit quantization (needs more VRAM)")
def fine_tune(base_model, dataset, output, epochs, batch_size, lr,
              max_seq_length, no_quantize):
    """Fine-tune a Llama model with LoRA."""

    config = load_training_config()

    print(f"\n{'='*60}")
    print(f"Fine-tuning: {base_model}")
    print(f"Dataset:     {dataset}")
    print(f"Output:      {output}")
    print(f"Epochs:      {epochs}")
    print(f"4-bit:       {not no_quantize}")
    print(f"{'='*60}\n")

    # ── Load tokenizer ──
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model ──
    print("Loading model...")
    model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    if not no_quantize:
        quant_config = config["quantization"]
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quant_config["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
        )

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if not no_quantize:
        model = prepare_model_for_kbit_training(model)

    # ── Apply LoRA ──
    print("Applying LoRA...")
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Load dataset ──
    print(f"Loading dataset from {dataset}...")
    ds = load_dataset("json", data_files=dataset, split="train")
    ds = ds.map(lambda x: {"text": format_example(x)})

    print(f"Training examples: {len(ds)}")
    print(f"Sample formatted:\n{ds[0]['text'][:300]}...\n")

    # ── Training ──
    training_args = TrainingArguments(
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=lr,
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        fp16=config["training"]["fp16"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        report_to="none",
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=training_args,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    print("Starting training...\n")
    trainer.train()

    # ── Save ──
    print(f"\nSaving model to {output}...")
    trainer.save_model(output)
    tokenizer.save_pretrained(output)

    print(f"\n{'='*60}")
    print(f"Done! Model saved to: {output}")
    print(f"\nNext steps:")
    print(f"  1. Merge LoRA weights (optional):")
    print(f"     Use the merge script or load with PEFT")
    print(f"  2. Quantize for local use:")
    print(f"     python scripts/quantize.py --model-path {output} --output models/my-model.gguf")
    print(f"{'='*60}")


if __name__ == "__main__":
    fine_tune()
