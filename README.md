# 1-Bit-Wonder

![](image.png)

This repository contains training and inference code supplementing the paper [1-Bit-Wonder: Improving QAT Performance in the Low-Bit Regime through K-Means Quantization](). We implement quantization-aware-training (QAT) with standard integer quantization as well as k-means based quantization which can be run via Torchtitan, as well as providing a Huggingface inference implementation that utilizes [kernels](https://github.com/graphcore-research/fused-dequantisation-kernels) optimized for k-means quantization.

## Installation

First, install the the package manager [uv](https://docs.astral.sh/uv/getting-started/installation/). Then create a virtual environment by running 

```bash
uv sync
```
from the root of the repository and activate it via `source .venv/bin/activate`.

Alternatively, if you are only interested in training, the subdirectory `src/llama3_qat/experiment` can be registered and run as a Torchtitan experiment, but this might not work seamlessly on the newest Torchtitan version, as this codebase is still heavily in development.

## Training

Our training code is built on top of [torchtitan](https://github.com/pytorch/torchtitan/tree/v0.2.1) (v0.2.1), for details regarding model training we refer to the documentation therein.

To start a training, activate the environment and run `CONFIG_FILE=/path/to/your/config bash run_train.sh`. If no config is specified, the training uses a small debug config.

The configs for the models from our main experiments are located in `train_configs`. Except for the debug model, all models were trained using the Llama 3.1 tokenizer, which can be dowloaded with [download_hf_assets.py](https://github.com/pytorch/torchtitan/blob/v0.2.1/scripts/download_hf_assets.py) from Torchtitan:

```bash
# Get your HF token from https://huggingface.co/settings/tokens

# Llama 3.1 tokenizer
python download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

Then replace `tokenizer_path` under the `model` section in the config with the path to the downloaded tokenizer.

For training on a different dataset than our small debug dataset we recommend to use the extension points that Torchtitan offers to implement a custom dataloader.

The specific QAT parameters are contained in the `Quantization` subconfig and quantized linear layers have to activated by choosing the model converter `"quantized_linear"`:

```toml
[model]
name = "llama3_qat" # specifies to use our experiment code
converters = ["quantized_linear"] # converts linear layers to quantized linear

[quantization] # specific quantization hyperparameters
quantizer = "nonlinear" # select "nonlinear" for k-means, "sym_int" for integer
target_bit_width = 4 # number of bits used
qat_start_step = 1000 # only applies QAT after this number of steps
buffer_update_interval = 10000 # interval at which the k-means centroids get updated
```

The rest of the config contains standard Torchtitan subconfigs and fields.

## Inference

Our inference is based on the Hugginface implementation of Llama3, patched with efficient [kernels](https://github.com/graphcore-research/fused-dequantisation-kernels) for k-means dequantization. To prepare a checkpoint after training for inference, use

```bash
python export_to_hf.py /your/checkpoint/after/training /output/path/for/inference/checkpoint
```
with additional command line args (e.g. `--bits 4`) to specify the quantization config.

To run a single-turn chat, you can then load the converted checkpoint with 

```bash
python chat.py --model_path /your/converted/checkpoint
```

or download the checkpoint directly from Huggingface by specifying `--hf_repo` instead. We provide the following checkpoints from the main experiments in our paper, which are already converted:

- [4B bfloat16](https://huggingface.co/Aleph-Alpha-Research/1BW-Llama-4B-16Bit-SFT)
- [12B 4-bit k-means quantized](https://huggingface.co/Aleph-Alpha-Research/1BW-Llama-12B-4Bit-SFT)
- [30B 1-bit k-means quantized](https://huggingface.co/Aleph-Alpha-Research/1BW-Llama-30B-1Bit-SFT)

All models use ~8GB of weight memory and can be run on consumer GPUs.

DISCLAIMER: All of the above checkpoints are trained from scratch, but use the Llama-3.1 tokenizer and are therefore built with Llama and subject to the Llama 3.1 community license agreement.

