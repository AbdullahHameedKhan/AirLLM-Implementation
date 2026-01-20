# AirLLM Inference Demo: Qwen2.5-14B

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-AirLLM-green)
![Hardware](https://img.shields.io/badge/Hardware-GPU%20Required-orange)

This repository contains a Jupyter/Colab notebook demonstrating how to run the **Qwen2.5-14B** model (specifically `JungZoona/T3Q-qwen2.5-14b-v1.0-e3`) using the **AirLLM** library.

## 🚀 Overview

Running large language models (10B+ parameters) typically requires high-end GPUs with significant VRAM. [AirLLM](https://github.com/lyogavin/airllm) enables the inference of these large models on consumer-grade GPUs (or free-tier Google Colab instances) by utilizing **layer-wise inference**. This technique loads the model layer-by-layer rather than loading the entire model into VRAM at once.

## 📋 Prerequisites

To run this notebook, you need a GPU-enabled environment (e.g., Google Colab with T4, L4, or A100).

### Dependencies
The following libraries are required and are installed within the notebook:

* `airllm`
* `transformers==4.46.0`
* `optimum==1.25.0`
* `accelerate`
* `bitsandbytes`

## 💻 Usage

1.  Clone this repository or open the notebook directly in Google Colab.
2.  Install the dependencies using the provided cells:
    ```bash
    !pip install airllm
    !pip install "transformers==4.46.0" "optimum==1.25.0" "accelerate" "bitsandbytes"
    ```
3.  Run the inference code.

### Code Snippet

The core logic uses `AirLLM.AutoModel` to handle the memory-efficient loading:

```python
from airllm import AutoModel

# Configuration
MAX_LENGTH = 128
MODEL_ID = "JungZoona/T3Q-qwen2.5-14b-v1.0-e3"

# Load model using AirLLM optimization
model = AutoModel.from_pretrained(MODEL_ID)

input_text = ['What is the capital of United States?']

# Tokenize
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False)
           
# Generate Output
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

# Decode
output = model.tokenizer.decode(generation_output.sequences[0])
print(output)
