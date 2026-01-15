# Gemma3-270M Swahili Fine-tuning

Fine-tuning Gemma3-270M, a small-sized Large Language Model (LLM), specifically adapted for Swahili language instruction-following and conversation tasks.

## Overview

This project demonstrates supervised fine-tuning of the Gemma3-270M model to understand and respond to Swahili instructions, enabling better conversational AI capabilities in the Swahili language. The fine-tuning process uses LoRA (Low-Rank Adaptation) for parameter-efficient training, making it memory-efficient and faster.

## Model on Hugging Face

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/ngusadeep/gemma-3-270M-Swahili-llm)

**Model**: [ngusadeep/gemma-3-270M-Swahili-llm](https://huggingface.co/ngusadeep/gemma-3-270M-Swahili-llm)

## Features

- **Model**: Gemma3-270M (270M parameters)
- **Language**: Swahili
- **Task Type**: Instruction-following and conversation-based fine-tuning
- **Training Method**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Dataset**: ~67,000 Swahili instruction-response pairs

## Technologies Used

- **Unsloth**: Fast and memory-efficient fine-tuning framework
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Pre-trained model library
- **LoRA/PEFT**: Parameter-efficient fine-tuning
- **KaggleHub**: Dataset management
- **TRL**: Training library for language models

## Dataset

The project uses the **Swahili Instructions** dataset from Kaggle:
- **Source**: [Swahili Instructions Dataset](https://www.kaggle.com/datasets/alfaxadeyembe/swahili-instructions/data)
- **Size**: ~67,000 instruction-response pairs
- **Format**: JSON with `instruction`, `input`, `output`, and `id` fields

## Installation

```bash
pip install unsloth transformers==4.56.2 trl==0.22.2 kagglehub datasets
```

## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ngusadeep/Gemma3-270M-Swahili/blob/main/notebooks/Gemma3_(270M)_Swahili_Finetuning.ipynb)

Open and run the Jupyter notebook for training:
```bash
jupyter notebook notebooks/Gemma3_(270M)_Swahili_Finetuning.ipynb
```

## Inference

### Using Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "ngusadeep/gemma-3-270M-Swahili-llm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prepare input
messages = [{"role": "user", "content": "Eleza nini maana ya uongozi."}]

# Apply chat template and generate
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    do_sample=True
)

# Decode response
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### Using Unsloth (Recommended)

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

# Load model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ngusadeep/gemma-3-270M-Swahili-llm",
    max_seq_length=2048,
    load_in_4bit=False,
)

# Set up chat template
tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

# Generate response
messages = [{"role": "user", "content": "Eleza nini maana ya uongozi."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).removeprefix('<bos>')

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
```

### Example Swahili Prompts

```python
"Eleza nini maana ya uongozi."  # Explanation
"Tunga hadithi fupi kuhusu safari."  # Creative writing
"Ni nini tofauti kati ya mchana na usiku?"  # Q&A
"Andika sentensi tano kuhusu elimu."  # Instruction following
```

### Recommended Parameters

- **temperature**: 1.0
- **top_p**: 0.95
- **top_k**: 64

## Training Configuration

- **LoRA Rank**: 128
- **Max Sequence Length**: 2048
- **Batch Size**: 4 per device
- **Learning Rate**: 5e-5
- **Optimizer**: AdamW 8-bit

## Results

After fine-tuning, the model demonstrates improved capability to:
- Understand Swahili instructions
- Generate appropriate responses in Swahili
- Follow conversational patterns
- Handle various instruction types

## Links

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/ngusadeep/gemma-3-270M-Swahili-llm)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ngusadeep/Gemma3-270M-Swahili/blob/main/notebooks/Gemma3_(270M)_Swahili_Finetuning.ipynb)
[![Unsloth Docs](https://img.shields.io/badge/Unsloth-Docs-blue)](https://docs.unsloth.ai/)

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google's Gemma3 model
- Unsloth team for the efficient fine-tuning framework
- Kaggle dataset contributors
