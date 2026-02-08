# 🦙 Llama Local — Run & Fine-Tune Llama on Your Machine

Run Meta's Llama 3.1 (8B or 70B) locally on your laptop, with optional cloud fine-tuning.

## Quick Start

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/llama-local.git
cd llama-local

# Install dependencies
pip install -r requirements.txt

# Download a model (8B quantized — works on 16GB RAM)
python scripts/download_model.py --model 8b

# Chat with it
python scripts/chat.py --model 8b
```

## What's Inside

```
llama-local/
├── scripts/
│   ├── download_model.py   # Download GGUF models from Hugging Face
│   ├── chat.py             # Interactive chat interface
│   ├── server.py           # REST API server (FastAPI)
│   ├── fine_tune.py        # Fine-tuning script (cloud GPU)
│   └── quantize.py         # Convert fine-tuned model to GGUF
├── configs/
│   ├── model_configs.json  # Model paths and settings
│   └── training_config.json# Fine-tuning hyperparameters
├── requirements.txt
├── requirements-training.txt
├── .gitignore
└── README.md
```

## System Requirements

| Model | RAM (CPU) | VRAM (GPU) | Speed |
|-------|-----------|------------|-------|
| 8B Q4 | 8GB+ | 6GB+ | ~20 tok/s GPU, ~8 tok/s CPU |
| 8B Q8 | 12GB+ | 10GB+ | ~15 tok/s GPU, ~5 tok/s CPU |
| 70B Q4 | 40GB+ | 2x 24GB | ~5 tok/s GPU, ~2 tok/s CPU |

## Model Downloads

The `download_model.py` script pulls quantized GGUF models from Hugging Face:

```bash
# Llama 3.1 8B (recommended for most laptops)
python scripts/download_model.py --model 8b

# Llama 3.1 8B higher quality quantization
python scripts/download_model.py --model 8b --quant Q8_0

# Llama 3.1 70B (needs 40GB+ RAM)
python scripts/download_model.py --model 70b
```

## Running Locally

### Interactive Chat
```bash
python scripts/chat.py --model 8b
python scripts/chat.py --model 70b
python scripts/chat.py --model 8b --system "You are a security analyst."
```

### API Server
```bash
# Start the server
python scripts/server.py --model 8b --port 8000

# Query it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## Fine-Tuning (Cloud GPU)

Fine-tuning requires a GPU. Use Google Colab (free T4), RunPod, Lambda, or any cloud GPU provider.

### 1. Prepare your training data

Create a JSONL file with your training examples:

```jsonl
{"instruction": "What is IDOR?", "response": "Insecure Direct Object Reference (IDOR) is a vulnerability where..."}
{"instruction": "How do you test for broken access control?", "response": "Testing for broken access control involves..."}
```

### 2. Run fine-tuning

```bash
# Install training dependencies (on cloud GPU machine)
pip install -r requirements-training.txt

# Fine-tune with LoRA (efficient — works on a single A100 or even T4 for 8B)
python scripts/fine_tune.py \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset your_data.jsonl \
  --output ./my-fine-tuned-model \
  --epochs 3
```

### 3. Quantize and run locally

```bash
# Convert to GGUF for local use
python scripts/quantize.py \
  --model-path ./my-fine-tuned-model \
  --output ./models/my-model.gguf \
  --quant Q4_K_M

# Chat with your fine-tuned model
python scripts/chat.py --model-path ./models/my-model.gguf
```

## Hugging Face Access

Llama 3.1 requires accepting Meta's license. Do this once:

1. Go to [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
2. Accept the license agreement
3. Create a Hugging Face token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Run `huggingface-cli login` and paste your token

## License

Code in this repo: MIT. Model weights are subject to [Meta's Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).
