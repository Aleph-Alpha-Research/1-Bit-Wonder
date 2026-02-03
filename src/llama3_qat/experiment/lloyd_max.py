from typing import Literal, Optional, TypeAlias, Union

import torch
from torch import Tensor


# ----------------------------- K-Means helpers ----------------------------- #


LloydMaxInit: TypeAlias = Union[
    Tensor,
    tuple[Literal["uniform_rms"], float],
    Literal["uniform_minmax", "random", "kmeans++", "cuberoot"],
]


@torch.no_grad()
def _lloyd_max_init(
    init: LloydMaxInit,
    tensor: Tensor,
    weight: Tensor | None,
    codepoints: int,
    generator: Optional[torch.Generator],
) -> Tensor:
    """From Orr et al., 2024, Optimal Formats for Weight Quantisation.
    https://github.com/graphcore-research/optimal-weight-formats
    """
    if isinstance(init, Tensor):
        assert init.shape == (codepoints,)
        return init.to(tensor.dtype, copy=True)
    if isinstance(init, tuple) and len(init) == 2 and init[0] == "uniform_rms":
        mean, std = tensor.mean(), tensor.std()
        return torch.linspace(
            mean - init[1] * std,
            mean + init[1] * std,
            codepoints,
            device=tensor.device,
            dtype=tensor.dtype,
        )
    if init == "uniform_minmax":
        return torch.linspace(
            tensor.min(),
            tensor.max(),
            codepoints,
            device=tensor.device,
            dtype=tensor.dtype,
        )
    if init == "random":
        return tensor[:codepoints].clone()  # already shuffled
    if init == "kmeans++":
        s = tensor[: int(2**20)]
        if s.numel() == 0:
            return torch.linspace(
                -1, 1, codepoints, device=tensor.device, dtype=tensor.dtype
            )
        midpoints = torch.empty(codepoints, device=s.device, dtype=s.dtype)
        p = torch.ones_like(s)
        for i in range(codepoints):
            probs = p
            total = probs.sum()
            j = torch.multinomial(probs / total, 1, generator=generator)
            midpoints[i] = s[j]
            midpoints[: i + 1] = midpoints[: i + 1].sort().values
            if i + 1 == codepoints:
                break
            boundaries = (midpoints[: i + 1][:-1] + midpoints[: i + 1][1:]) / 2
            closest = torch.bucketize(s, boundaries)
            p = (s - midpoints[closest]) ** 2
            if weight is not None:
                p = p * weight[: len(s)].pow(2)
        return midpoints

    if init == "cuberoot":
        # Note: doesn't respect `weight`
        s = tensor[: int(2**20)].sort().values
        delta = (s[1:] - s[:-1]) ** (2 / 3)
        # delta += delta.mean()
        delta_sum = delta.cumsum(0)
        loc = torch.linspace(
            0, delta_sum[-1], codepoints + 2, device=s.device, dtype=s.dtype
        )[1:-1]
        # Note - it would be better to interpolate here, rather than round-to-nearest
        return s[torch.bucketize(loc, delta_sum)]
    raise ValueError(f"Unexpected init scheme {init}")


@torch.no_grad()
def lut_lloyd_max(
    tensor: Tensor,
    bits: float | None,
    *,
    codepoints: Optional[int] = None,
    threshold: float = 1e-4,
    weight: Optional[Tensor] = None,
    init: LloydMaxInit = "kmeans++",
    incremental: bool = True,
    max_samples: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """From Orr et al., 2024, Optimal Formats for Weight Quantisation.
    https://github.com/graphcore-research/optimal-weight-formats

    Use Lloyd-Max (k-means) to find the RMS-optimal quantiser for the given tensor.

    threshold -- when the ratio of changed cluster assignments <= threshold, stop

    weight -- if provided, a positive tensor the same shape as `tensor`, to use as an
              importance weight for each sample

    incremental -- start with a subset of the data and scale up
    """

    # Preparation: shuffle, truncate, cast, get init
    # Resolve number of centroids
    if codepoints is None:
        if bits is None:
            raise ValueError("Provide either codepoints or bits.")
        codepoints = int(round(2 ** float(bits)))
    codepoints = int(codepoints)

    # If tensor is empty or numerically constant, return a simple grid to avoid degenerate init
    if tensor.numel() == 0:
        return torch.linspace(
            -1, 1, codepoints, device=tensor.device, dtype=tensor.dtype
        )

    idx = torch.randperm(
        tensor.nelement(), device=tensor.device, dtype=torch.int32, generator=generator
    )

    tensor = tensor.flatten()[idx]
    if weight is not None:
        weight = weight.flatten().to(tensor.device)[idx]

    if max_samples is not None:
        tensor = tensor[:max_samples]
    if dtype is None:
        # Very large tensors have stability problems due the float32
        # mantissa length, so default to float64
        dtype = torch.float32 if tensor.nelement() <= 2**26 else torch.float64
    tensor = tensor.to(dtype)
    if weight is not None:
        weight = weight.to(dtype)

    midpoints = _lloyd_max_init(init, tensor, weight, codepoints, generator)

    if weight is not None:
        sum_weight = torch.empty_like(midpoints)

    # K-means iteration
    assign = torch.empty(tensor.shape, device=tensor.device, dtype=torch.int64)
    last_assign = torch.empty_like(assign)
    n = 2**20 if incremental else tensor.nelement()

    for _ in range(1000):
        n = min(n, tensor.nelement())
        last_assign[:n] = assign[:n]
        boundaries = (midpoints[1:] + midpoints[:-1]) / 2
        torch.bucketize(tensor[:n], boundaries, out=assign[:n])

        if weight is None:
            midpoints.scatter_reduce_(
                0, assign[:n], tensor[:n], "mean", include_self=False
            )
        else:
            midpoints.scatter_reduce_(
                0, assign[:n], tensor[:n] * weight[:n], "sum", include_self=False
            )
            sum_weight.zero_().scatter_reduce_(
                0, assign[:n], weight[:n], "sum", include_self=False
            )
            midpoints.div_(sum_weight.clamp_min_(torch.finfo(dtype).smallest_normal))

        midpoints = torch.cummax(midpoints, 0).values
        change = (last_assign[:n] != assign[:n]).float().mean().item()
        if change <= threshold:
            if tensor.nelement() <= n:
                break
            n *= 2
    assert (midpoints[:-1] <= midpoints[1:]).all().item()
    return midpoints

