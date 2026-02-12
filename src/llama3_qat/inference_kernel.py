import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available, using torch.compile fallback")


def build_lut8(centroids: torch.Tensor, bits: int) -> torch.Tensor:
    """Build 256-entry LUT from centroids for packed byte format."""
    elements_per_byte = 8 // bits
    return torch.cartesian_prod(*([centroids] * elements_per_byte)).view(
        256, elements_per_byte
    )


_LUT_CACHE: dict[tuple, torch.Tensor] = {}


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config(dict(BLOCK_SIZE=256), num_stages=1, num_warps=1),
            triton.Config(dict(BLOCK_SIZE=512), num_stages=1, num_warps=1),
            triton.Config(dict(BLOCK_SIZE=1024), num_stages=1, num_warps=1),
        ],
        key=["k", "ELEMENTS_PER_BYTE", "GROUP_SIZE"],
    )
    @triton.jit
    def _mv_lut_kernel(
        a_ptr,
        b_ptr,
        lut_ptr,
        bs_ptr,
        out_ptr,
        k,
        ELEMENTS_PER_BYTE: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        """Matrix-vector LUT kernel for batch_size=1."""
        BLOCK_ELEMENTS: tl.constexpr = BLOCK_SIZE * ELEMENTS_PER_BYTE
        tl.static_assert(BLOCK_ELEMENTS % GROUP_SIZE == 0)
        BLOCK_GROUPS: tl.constexpr = BLOCK_ELEMENTS // GROUP_SIZE
        n = tl.program_id(axis=0)

        offs_a = tl.arange(0, BLOCK_ELEMENTS)
        offs_b = tl.arange(0, BLOCK_SIZE)
        offs_bs = tl.arange(0, BLOCK_GROUPS)
        a_ptrs = a_ptr + offs_a
        b_ptrs = b_ptr + n * (k // ELEMENTS_PER_BYTE) + offs_b
        bs_ptrs = bs_ptr + n * (k // GROUP_SIZE) + offs_bs
        out = tl.zeros((), dtype=tl.float32)

        for ik in range(0, tl.cdiv(k, BLOCK_ELEMENTS)):
            a = tl.load(a_ptrs, mask=offs_a + ik * BLOCK_ELEMENTS < k, other=0.0)
            bq = tl.load(
                b_ptrs,
                mask=offs_b + ik * BLOCK_SIZE < (k // ELEMENTS_PER_BYTE),
                other=0,
            )
            bu = tl.reshape(
                tl.load(
                    lut_ptr
                    + ELEMENTS_PER_BYTE * tl.cast(bq[:, None], tl.int32)
                    + tl.arange(0, ELEMENTS_PER_BYTE),
                ),
                [BLOCK_ELEMENTS],
            )
            bs = tl.load(
                bs_ptrs, mask=offs_bs + ik * BLOCK_GROUPS < (k // GROUP_SIZE), other=0.0
            )
            b = tl.reshape(
                tl.reshape(bu, [BLOCK_GROUPS, GROUP_SIZE]) * bs[:, None], BLOCK_ELEMENTS
            )
            if ELEMENTS_PER_BYTE >= 4:
                out += tl.sum(a * b, dtype=tl.float32)
            else:
                out += tl.sum(a.to(tl.float32) * b.to(tl.float32))

            a_ptrs += BLOCK_ELEMENTS
            b_ptrs += BLOCK_SIZE
            bs_ptrs += BLOCK_GROUPS

        tl.store(out_ptr + n, out)

    @triton.autotune(
        configs=[
            triton.Config(
                dict(
                    BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=128, GROUP_SIZE_M=8
                ),
                num_stages=1,
                num_warps=1,
            ),
            triton.Config(
                dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64, GROUP_SIZE_M=8),
                num_stages=1,
                num_warps=1,
            ),
            triton.Config(
                dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=8),
                num_stages=1,
                num_warps=1,
            ),
            triton.Config(
                dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=8, BLOCK_SIZE_K=256, GROUP_SIZE_M=8),
                num_stages=1,
                num_warps=2,
            ),
        ],
        key=["m", "n", "k", "ELEMENTS_PER_BYTE"],
    )
    @triton.jit
    def _mm_lut_kernel(
        a_ptr,
        b_ptr,
        lut_ptr,
        bs_ptr,
        out_ptr,
        m,
        n,
        k,
        ELEMENTS_PER_BYTE: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ) -> None:
        """Matrix-matrix LUT kernel for batch_size>1."""
        BLOCK_K_ELEMENTS: tl.constexpr = BLOCK_SIZE_K * ELEMENTS_PER_BYTE
        tl.static_assert(BLOCK_K_ELEMENTS % GROUP_SIZE == 0)
        BLOCK_K_GROUPS: tl.constexpr = BLOCK_K_ELEMENTS // GROUP_SIZE

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(m > 0)
        tl.assume(n > 0)
        tl.assume(k > 0)

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % n
        offs_ak = tl.arange(0, BLOCK_K_ELEMENTS)
        offs_bk = tl.arange(0, BLOCK_SIZE_K)
        offs_bsk = tl.arange(0, BLOCK_K_GROUPS)
        offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        a_ptrs = a_ptr + (offs_am[:, None] * k + offs_ak[None, :])
        b_ptrs = b_ptr + (
            offs_bn[:, None] * (k // ELEMENTS_PER_BYTE) + offs_bk[None, :]
        )
        bs_ptrs = bs_ptr + (offs_bn[:, None] * (k // GROUP_SIZE) + offs_bsk[None, :])
        out_ptrs = out_ptr + offs_outm[:, None] * n + offs_outn[None, :]

        out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ik in range(0, tl.cdiv(k, BLOCK_K_ELEMENTS)):
            a = tl.reshape(
                tl.load(
                    a_ptrs, mask=offs_ak[None, :] < k - ik * BLOCK_K_ELEMENTS, other=0.0
                ),
                [BLOCK_SIZE_M, BLOCK_K_GROUPS, GROUP_SIZE],
            )
            qb = tl.load(b_ptrs)
            bu = tl.reshape(
                tl.load(
                    lut_ptr
                    + ELEMENTS_PER_BYTE * tl.cast(qb[:, :, None], tl.int32)
                    + tl.arange(0, ELEMENTS_PER_BYTE),
                ),
                [BLOCK_SIZE_N, BLOCK_K_GROUPS, GROUP_SIZE],
            )
            bs = tl.load(bs_ptrs)
            ab = tl.dot(tl.trans(a, 1, 0, 2), tl.trans(bu, 1, 2, 0))
            out = out + tl.sum(ab * tl.trans(bs)[:, None, :], axis=0)

            a_ptrs += BLOCK_K_ELEMENTS
            b_ptrs += BLOCK_SIZE_K
            bs_ptrs += BLOCK_K_GROUPS

        tl.store(
            out_ptrs, out, mask=(offs_outm[:, None] < m) & (offs_outn[None, :] < n)
        )


def _run_mv_lut_triton(a, b, lut, bs, out):
    (k,) = a.shape
    (n,) = out.shape
    elements_per_byte = lut.shape[1]
    group_size = k // bs.shape[1]
    _mv_lut_kernel[(n,)](
        a,
        b,
        lut,
        bs,
        out,
        k,
        ELEMENTS_PER_BYTE=elements_per_byte,
        GROUP_SIZE=group_size,
    )


def _run_mm_lut_triton(a, b, lut, bs, out):
    m, k = a.shape
    _, n = out.shape
    elements_per_byte = lut.shape[1]
    group_size = k // bs.shape[1]
    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )
    _mm_lut_kernel[grid](
        a,
        b,
        lut,
        bs,
        out,
        m,
        n,
        k,
        ELEMENTS_PER_BYTE=elements_per_byte,
        GROUP_SIZE=group_size,
    )


@torch.compile(mode="max-autotune-no-cudagraphs")
def _mm_lut_compiled(a, bq, lut8, bs):
    """Fallback LUT matmul with torch.compile."""
    bv = lut8[bq.long()].flatten(start_dim=1)
    b = bv.view(*bs.shape, -1).mul(bs[:, :, None]).flatten(start_dim=1)
    return torch.matmul(a, b.T)


def mm_lut(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    centroids: torch.Tensor,
    scales: torch.Tensor,
    bits: int = 4,
    block_size: int = 64,
) -> torch.Tensor:
    """Matrix multiplication with LUT-based dequantization."""
    global _LUT_CACHE

    use_triton = TRITON_AVAILABLE and x.is_cuda

    batch_dims = x.shape[:-1]
    in_features = x.shape[-1]
    out_features = scales.shape[0]

    x_flat = x.reshape(-1, in_features).contiguous()
    m = x_flat.shape[0]

    # Build LUT8 (cached)
    cache_key = (id(centroids), centroids.device, bits)
    if cache_key not in _LUT_CACHE:
        _LUT_CACHE[cache_key] = build_lut8(centroids, bits).to(x.device, x.dtype)
    lut8 = _LUT_CACHE[cache_key]

    weight_packed = weight_packed.contiguous()
    scales = scales.to(x.dtype).contiguous()

    if use_triton:
        out = torch.empty((m, out_features), device=x.device, dtype=x.dtype)
        if m == 1:
            _run_mv_lut_triton(x_flat[0], weight_packed, lut8, scales, out[0])
        else:
            _run_mm_lut_triton(x_flat, weight_packed, lut8, scales, out)
    else:
        out = _mm_lut_compiled(x_flat, weight_packed, lut8, scales)

    return out.reshape(*batch_dims, out_features)


class QuantizedLinearTriton(torch.nn.Module):
    """Quantized linear layer using Triton LUT kernels for fast inference."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        block_size: int = 64,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size
        self.num_centroids = 2**bits

        elements_per_byte = 8 // bits
        packed_in = in_features // elements_per_byte
        num_blocks = in_features // block_size

        # Buffers for quantized weights
        self.register_buffer(
            "weight_packed",
            torch.zeros(out_features, packed_in, dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "scales", torch.zeros(out_features, num_blocks, dtype=dtype, device=device)
        )
        self.register_buffer(
            "centroids",
            torch.zeros(self.num_centroids, dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Triton mm_lut kernel."""
        # x: [batch, seq, in_features] or [seq, in_features]
        orig_shape = x.shape
        if x.dim() == 3:
            batch, seq, _ = x.shape
            x = x.view(batch * seq, -1)
        elif x.dim() == 1:
            x = x.unsqueeze(0)

        # mm_lut expects centroids (not pre-built LUT), it builds LUT internally
        out = mm_lut(
            x.contiguous(),
            self.weight_packed,
            self.centroids,  # Pass centroids, mm_lut builds LUT
            self.scales.to(x.dtype),
            bits=self.bits,
            block_size=self.block_size,
        )

        # Reshape back
        if len(orig_shape) == 3:
            out = out.view(batch, seq, -1)
        elif len(orig_shape) == 1:
            out = out.squeeze(0)

        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}, block={self.block_size}"
