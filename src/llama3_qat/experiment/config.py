from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Quantization:
    quantizer: Literal["sym_int", "ternary", "binary", "nonlinear"] = "sym_int"

    ignore_names: tuple[str, ...] = ("output",)

    target_bit_width: int = 8

    block_dim_1: int | None = 1

    block_dim_2: int | None = 64

    relative_scale: Literal["none", "absmax", "absmean", "rms"] = "absmax"

    centroid_update_algorithm: Literal["kmeans", "fischer_kmeans"] = "kmeans"

    buffer_update_interval: int | None = None  # disable by default

    qat_start_step: int = 0


@dataclass
class JobConfig:
    quantization: Quantization = field(default_factory=Quantization)
