# InfLLM: Unveiling the Intrinsic Capacity of LLMs for Understanding Extremely Long Sequences with Training-Free Memory

The code of our paper "InfLLM: Unveiling the Intrinsic Capacity of LLMs for Understanding Extremely Long Sequences with Training-Free Memory" [[pdf]](https://arxiv.org/pdf/2402.04617.pdf).

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
```
bash scripts/chat.sh
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



