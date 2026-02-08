# 🦙 Llama Local & Cloud Setup Guide

A complete walkthrough for running Meta's Llama 3.1 (8B and 70B) on **Windows**, **Linux**, **locally**, and in the **cloud**.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Model Comparison](#model-comparison)
- [Local Setup — Linux](#local-setup--linux)
- [Local Setup — Windows](#local-setup--windows)
- [Cloud Setup — RunPod](#cloud-setup--runpod)
- [Cloud Setup — Google Colab (Free)](#cloud-setup--google-colab-free)
- [Cloud Setup — AWS / GCP / Azure](#cloud-setup--aws--gcp--azure)
- [Running as an API Server](#running-as-an-api-server)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before anything, you need access to the Llama models:

1. Go to [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
2. Click **"Agree and access repository"** to accept Meta's license
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Save the token — you'll need it below

---

## Model Comparison

| | Llama 3.1 8B | Llama 3.1 70B |
|---|---|---|
| **Best for** | Laptops, desktops, quick tasks | Workstations, servers, complex tasks |
| **RAM needed (CPU)** | 8–12 GB | 40–48 GB |
| **VRAM needed (GPU)** | 6–10 GB | 2x 24 GB or 1x 48 GB |
| **Download size** | ~4.5 GB (Q4) | ~40 GB (Q4) |
| **Speed (CPU)** | ~5–10 tokens/sec | ~1–3 tokens/sec |
| **Speed (GPU)** | ~15–30 tokens/sec | ~5–15 tokens/sec |
| **Quality** | Good for most tasks | Best open-source quality |

**Recommendation:** Start with 8B. Move to 70B only if you need higher quality and have the hardware.

---

## Local Setup — Linux

### Option A: Ollama (Easiest)

[Ollama](https://ollama.com) handles everything — download, quantization, serving.

```bash
# Install ollama
curl -fsSL https://ollama.com/install.sh | sh

# Run 8B (downloads automatically on first run)
ollama run llama3.1

# Run 70B
ollama run llama3.1:70b

# Run as a background API server
ollama serve &
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.1",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

### Option B: llama.cpp (More Control)

```bash
# Install
pip install llama-cpp-python huggingface-hub

# If you have an NVIDIA GPU, install with CUDA support instead:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Log in to Hugging Face
huggingface-cli login
# Paste your token when prompted

# Download 8B model
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --local-dir ./models

# Download 70B model
huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
  Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  --local-dir ./models

# Run it (Python)
python3 -c "
from llama_cpp import Llama
llm = Llama(model_path='./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', n_gpu_layers=-1)
output = llm('What is the capital of France?', max_tokens=100)
print(output['choices'][0]['text'])
"
```

### Option C: vLLM (Fastest for GPUs)

Best throughput if you have a dedicated GPU.

```bash
pip install vllm

# Serve 8B
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000

# Serve 70B (needs ~2x A100 or similar)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --port 8000

# Query it (OpenAI-compatible)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## Local Setup — Windows

### Option A: Ollama (Easiest)

1. Download the installer from [ollama.com/download/windows](https://ollama.com/download/windows)
2. Run the installer
3. Open **Command Prompt** or **PowerShell**:

```powershell
# Run 8B
ollama run llama3.1

# Run 70B
ollama run llama3.1:70b
```

That's it. Ollama handles GPU detection automatically.

### Option B: LM Studio (GUI — No Terminal Needed)

1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Open LM Studio
3. Search for "Llama 3.1 8B Instruct GGUF" in the model browser
4. Click Download
5. Go to the Chat tab and start talking

For 70B, search "Llama 3.1 70B Instruct GGUF" — only download if you have 40GB+ RAM.

### Option C: llama.cpp on Windows

```powershell
# Install Python 3.10+ from python.org first

# Install llama-cpp-python
pip install llama-cpp-python huggingface-hub

# For NVIDIA GPU support (requires CUDA toolkit installed):
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Log in to Hugging Face
huggingface-cli login

# Download 8B
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF `
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf `
  --local-dir .\models

# Download 70B
huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF `
  Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf `
  --local-dir .\models

# Test it
python -c "from llama_cpp import Llama; llm = Llama(model_path='./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', n_gpu_layers=-1); print(llm('Hello!', max_tokens=50)['choices'][0]['text'])"
```

### Option D: WSL2 (Run the Linux instructions on Windows)

```powershell
# Enable WSL
wsl --install

# Open Ubuntu from Start Menu, then follow the Linux instructions above
```

---

## Cloud Setup — RunPod

[RunPod](https://runpod.io) gives you on-demand GPU servers. Good for 70B.

### 1. Rent a GPU

| Model | Recommended GPU | Cost (approx) |
|-------|----------------|----------------|
| 8B | 1x A40 or L4 | ~$0.40/hr |
| 70B | 2x A100 80GB | ~$3.50/hr |

### 2. Deploy with a template

RunPod has one-click templates:
- Go to **Pods → Deploy**
- Search for **"vLLM"** or **"Ollama"** template
- Select your GPU
- Set the model name as an environment variable:
  ```
  MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
  ```
- Deploy

### 3. Or deploy manually via SSH

```bash
# SSH into your RunPod instance
ssh root@<your-pod-ip>

# Install and run with ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama run llama3.1:70b

# Or use vLLM for an API
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000
```

### 4. Access from your local machine

```bash
# If using ollama on RunPod
curl http://<pod-ip>:11434/api/chat -d '{
  "model": "llama3.1:70b",
  "messages": [{"role": "user", "content": "Hello"}]
}'

# If using vLLM on RunPod
curl http://<pod-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## Cloud Setup — Google Colab (Free)

Works for 8B only (free tier gives you a T4 with 16GB VRAM).

Create a new notebook at [colab.research.google.com](https://colab.research.google.com) and paste:

```python
# Cell 1: Install
!pip install llama-cpp-python huggingface-hub -q

# Cell 2: Download 8B
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
)
print(f"Model at: {model_path}")

# Cell 3: Load and chat
from llama_cpp import Llama
llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096)

def chat(message):
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    output = llm(prompt, max_tokens=512, stop=["<|eot_id|>"])
    return output["choices"][0]["text"]

# Cell 4: Try it
print(chat("Explain quantum computing in simple terms"))
```

To run 70B on Colab, you need **Colab Pro** with an A100 runtime. Select **Runtime → Change runtime type → A100 GPU**.

---

## Cloud Setup — AWS / GCP / Azure

### AWS (EC2 + SageMaker)

```bash
# Launch a g5.2xlarge (1x A10G, 24GB VRAM) for 8B
# Launch a p4d.24xlarge (8x A100) for 70B

# SSH in, then:
sudo apt update && sudo apt install -y python3-pip
pip install vllm
huggingface-cli login

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000
```

### GCP (Compute Engine)

```bash
# Use a g2-standard-8 (1x L4) for 8B
# Use a a2-ultragpu-2g (2x A100) for 70B

# Same setup commands as AWS above
```

### Azure (NC-series VMs)

```bash
# Use Standard_NC8as_T4_v3 for 8B
# Use Standard_ND96amsr_A100_v4 for 70B

# Same setup commands as above
```

### All clouds — quick Docker option

```bash
# Pull and run vLLM container (works on any cloud with NVIDIA GPUs)
docker run --gpus all \
  -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN=<your-token> \
  vllm/vllm-openai \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

---

## Running as an API Server

No matter where you host (local or cloud), you can serve Llama as an OpenAI-compatible API.

### With Ollama

```bash
ollama serve
# API available at http://localhost:11434
```

### With vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000
# API available at http://localhost:8000
```

### Query from any language

```bash
# curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
# Python (works with openai library)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

```javascript
// JavaScript
const response = await fetch("http://localhost:8000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages: [{ role: "user", content: "Hello!" }],
  }),
});
const data = await response.json();
console.log(data.choices[0].message.content);
```

```powershell
# PowerShell
$body = @{
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    messages = @(@{role = "user"; content = "Hello!"})
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" `
  -Method POST -ContentType "application/json" -Body $body
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `n_gpu_layers` or use a smaller quantization (Q4 instead of Q8) |
| `Not enough RAM` for 70B | You need 40GB+ free RAM. Try 8B instead, or use cloud |
| `huggingface-cli login` fails | Make sure your token has "Read" permissions |
| Slow on CPU | Expected — CPU inference is 3-5x slower. Use GPU or reduce context size (`n_ctx=2048`) |
| `llama-cpp-python` won't install on Windows | Install Visual Studio Build Tools first: [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| Ollama can't find GPU on Windows | Update your NVIDIA drivers to latest. Restart after install |
| vLLM not working on Windows | vLLM is Linux-only. Use WSL2 or ollama on Windows |
| Model downloads stuck | Use `--resume-download` flag with huggingface-cli, or check disk space |

---

## License

Code in this repo: MIT. Model weights are subject to [Meta's Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).
