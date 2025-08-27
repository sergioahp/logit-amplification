# SPAR Diff Amplification

Sparse differential amplification

## Background

## Prerequisites

This project uses Llama 3.1 models which require access approval:

1. Request access to `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct` on [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B)
2. Login with your Hugging Face token: `huggingface-cli login`

## Installation

The correct PyTorch version depends on your hardware. Choose one:

### CUDA 12.1 (recommended for most GPUs)
```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8 (for older GPUs)
```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

### CPU only
```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

## Usage

Run the amplification script:
```bash
python src/logit_amplification.py
```

## On Vast
Provide setup.sh as the provisioning script to vast's torch container
