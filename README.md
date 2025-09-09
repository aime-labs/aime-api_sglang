# AIME API SGLang Worker

 [SGLang](https://docs.sglang.ai/) worker for [AIME API Server](https://github.com/aime-team/aime-api-server) to scalable serve large language models on CUDA or ROCM with various quantizations and large context lengths.

Supported models:

- Llama / Deepseek / Qwen / Mistral / Mixtral / DeepSeek / Gemma / InternLM / Phi /

For a full list of current supported models see [here](https://docs.sglang.ai/supported_models/generative_models.html)


## How to setup a AIME API SGLang worker with MLC

### Setup SGLang

(Setup with uv pip install didn't work in my case)

```bash
mlc create sglang Pytorch 2.8.0-aime 
mlc open sglang

git clone -b v0.5.1 https://github.com/sgl-project/sglang

cd sglang

pip install --upgrade pip
pip install -e "python[all]"

sudo apt install libnuma1

```
### Setup AIME SGLang Worker

```bash
git clone https://github.com/aime-labs/aime-api_sglang
cd aime-api_sglang
pip install -r requirements.txt

```

### Download LLM models

The installed AIME worker interface pip provides the 'awi' command to download model weights:

```bash
awi download-weights {model name} -o /path/to/your/model/weights/
```

e.g. to download LLama3.1 70B fp8 Instruct model:

```bash
awi download-weights meta-llama/Llama-3.3-70B-Instruct -o /path/to/your/model/weights/
```

## How to start a AIME API SGLang worker

### Running an LLM

```bash
python main.py --api_server http://127.0.0.1:7777 --api_auth_key b07e305b50505ca2b3284b4ae5f65d1 --model-path /path/to/your/model/weights/your_llm/ --job_type job_type_name --max_batch_size 8 --tensor-parallel-size 2
```

### Running Llama 3.3

Starting Llama 3.3 70B worker on 4x RTX 6000 Ada 48GB GPUs:

```bash
python main.py --api_server http://127.0.0.1:7777 --api_auth_key b07e305b50505ca2b3284b4ae5f65d1 --model-path /path/to/your/model/weights/Llama-3.3-70B-Instruct --job_type llama3 --max_batch_size 8 --tensor-parallel-size 4
```

### Adjust maximum context length


Starting Llama 3.3 70B worker on 2x RTX 6000 Ada 48GB GPUs with a maximum context length 8192 tokens:

```bash
python main.py --api_server http://127.0.0.1:7777 --api_auth_key b07e305b50505ca2b3284b4ae5f65d1 --model-path /path/to/your/model/weights/Llama-3.3-70B-Instruct --job_type llama3 --max_batch_size 8 --tensor-parallel-size 2 --context-length 8192
```

Copyright (c) AIME GmbH and affiliates. Find more info at https://www.aime.info/api

This software may be used and distributed according to the terms of the MIT LICENSE
