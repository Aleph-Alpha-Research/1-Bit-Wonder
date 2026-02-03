# 1-Bit-Wonder

## Quick start

First, install the the package manager [uv](https://docs.astral.sh/uv/getting-started/installation/). Then create a virtual environment by running

```bash
uv sync
```
from the root of the repository and activate it via `source .venv/bin/activate`.

Our training code is built on top of [torchtitan](https://github.com/pytorch/torchtitan/tree/v0.2.1) (v0.2.1), for details regarding training configuration we refer to the references within.

The specific QAT parameters are contained in the `Quantization` subconfig and quantized linear layers have to activated by choosing the model converter `"quantized_linear"` in the toml file:

```toml
[model]
name = "llama3_qat" # specifies to use our experiment code
converters = ["quantized_linear"] # converts linear layers to quantized linear

[quantization] # specific quantization hyperparameters
quantizer = "nonlinear"
target_bit_width = 4
qat_start_step = 5
buffer_update_interval = 2
```
To start a small training, activate the environment and run `bash run_train.sh`, which defaults to a small debug config that runs locally. 




