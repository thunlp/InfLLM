# InfLLM: Unveiling the Intrinsic Capacity of LLMs for Understanding Extremely Long Sequences with Training-Free Memory

The code of our paper "InfLLM: Unveiling the Intrinsic Capacity of LLMs for Understanding Extremely Long Sequences with Training-Free Memory" [[pdf]](https://arxiv.org/pdf/2402.04617.pdf).

## Updates
  - **March 3, 2024**: Initial code release. See [init](https://github.com/thunlp/InfLLM/tree/init).
  - **March 24, 2024**: Refactor the code. Improve inference speed and reduce GPU memory usage.
  - **April 4, 2024**: Supports topk retrieval using [faiss](https://github.com/facebookresearch/faiss).
  - **April 20, 2024**: Added support for LLaMA 3.

## Quick Links
* [Overview](#overview)
* [Requirements](#requirements)
* [Usage](#usage)
* [Citation](#citation)

## Overview

![overview](image/framework.png)

Large language models (LLMs) have emerged as a cornerstone in real-world applications with lengthy streaming inputs, such as LLM-driven agents. However, existing LLMs, pre-trained on sequences with restricted maximum length, cannot generalize to longer sequences due to the out-of-domain and distraction issues. To alleviate these issues, existing efforts employ sliding attention windows and discard distant tokens to achieve the processing of extremely long sequences. Unfortunately, these approaches inevitably fail to capture long-distance dependencies within sequences to deeply understand semantics. This paper introduces a training-free memory-based method, InfLLM, to unveil the intrinsic ability of LLMs to process streaming long sequences. Specifically, InfLLM stores distant contexts into additional memory units and employs an efficient mechanism to lookup token-relevant units for attention computation. Thereby, InfLLM allows LLMs to efficiently process long sequences while maintaining the ability to capture long-distance dependencies. Without any training, InfLLM enables LLMs pre-trained on sequences of a few thousand tokens to achieve superior performance than competitive baselines continually training these LLMs on long sequences. Even when the sequence length is scaled to 1, 024K, InfLLM still effectively captures long-distance dependencies.


## Requirements
```
torch>=1.13.1
transformers>=4.37.2
fschat>=0.2.35
datasets>=2.17.0
omegaconf
flash-attn

rouge==1.0.1
fuzzywuzzy==0.18.0
jieba==0.42.1
```

## Usage

### Configuration

We use YAML files for configuration, and you can see the configuration files we use for benchmark in the `config/` directory. 

The description of the configuration files is as follows:

```yaml
model: 
  # attention type. 
  # inf-llm/infinite-lm/stream-lm/origin(full attention)
  type: inf-llm 

  # huggingface or model-center model path
  path: mistralai/Mistral-7B-Instruct-v0.2 

  # Use flash-attention or not. 
  # For inf-llm/infinite-lm/stream-llm, we implemented multi-stage flash-attention by OpenAI's Triton.
  fattn: false 
  
  # RoPE base and distance_scale
  base: 1000000
  distance_scale: 1.0

  # inf-llm/infinite-lm/stream-lm settings

  # Initital tokens as attention sinks
  n_init: 128   
  # Local sliding window size
  n_local: 4096 

  # inf-llm settings

  # Number of memory units to retrieve for attention computation.
  topk: 16  
  # The number of top-scoring tokens per memory unit considered as representative elements. 
  repr_topk: 4 
  # Maximum number of memory units stored in GPU memory. 
  max_cached_block: 32
  # Number of tokens queried at a time as an execution block.
  # Each execution block retrieves topk memory units once.
  exc_block_size: 512
  
  # The strategy for replacing cached memory units. 
  # Supported strategies include LRU (Least Recently Used), FIFO (First In, First Out), 
  # and LRU-S (LRU in our paper).
  cache_strategy: lru

  # score_decay for LRU-S
  # score_decay: 0.1

  # Use overlap local and global calculation.
  # Can accelerate, but may not be compatible.
  async_global_stream: false

  # Use faiss for topk retrieval of memory units. 
  # It will increase inference time and ensure constant GPU memory usage.
  faiss: false 

  # Use perhead topk. 
  # Enabling it will be very time-consuming and is intended for research use only.
  # perhead: false

# Model max input length.
# A truncation will be employed if the input length exceeds.
max_len: 2147483647

# truncation type. Now supports suffix only.
truncation: suffix

# Chunked input in decoding.
# To save GPU memory. (FFN block)
chunk_size: 8192

# Conversation type. 
# mistral-inst/vicuna/qwen/minicpm/llama-3-inst
conv_type: mistral-inst
```

### Evaluation

**Data Preparation**
We adopt InfiniteBench and LongBench for model evaluation. You can download the datasets by running the following command.
```
bash scripts/download.sh
```

**Response Generation**
You can evaluate InfLLM by running the following command. Notably, the provided code is used to run evaluate with only one GPU, and you can accelerate the experiments with multiple GPUs.
```
bash scripts/[infinitebench,longbench].sh
```

### Run a Chatbot with InfLLM

We integrated fastchat's CLI chat.

```
python -m inf_llm.chat \
    --model-path mistralai/Mistral-7B-Instruct-v0.2 \
    --inf-llm-config-path config/mistral-inf-llm.yaml
```

## Citation
If you find InfLLM useful, please cite the following paper:
```
@article{xiao2024infllm,
  author       = {Chaojun Xiao and Pengle Zhang and Xu Han and Guangxuan Xiao and Yankai Lin and Zhengyan Zhang and Zhiyuan Liu and Song Han and Maosong Sun},
  title        = {InfLLM: Unveiling the Intrinsic Capacity of LLMs for Understanding
                  Extremely Long Sequences with Training-Free Memory},
  journal      = {arXiv},
  year         = {2024}
}
```



