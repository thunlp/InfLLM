# https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import math
import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, 
                    K_block_ptr, V_block_ptr, MASK_block_ptr, Logits_block_ptr,
                    start_m, qk_scale,  
                    N_CTX,
                    sliding_window_offset, sliding_window_size,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, SLIDING_WINDOW: tl.constexpr,
                    IS_EVEN_M: tl.constexpr, IS_EVEN_N: tl.constexpr, 
                    OUTPUT_LOGITS: tl.constexpr
                ):
    # range of values handled by this stage
    if SLIDING_WINDOW:
        lo = ((start_m * BLOCK_M + sliding_window_offset - sliding_window_size + 1) // BLOCK_N) * BLOCK_N
        hi = ((((start_m + 1) * BLOCK_M - 1) + sliding_window_offset + BLOCK_N) // BLOCK_N) * BLOCK_N
        if lo < 0:
            lo = 0
        if hi > N_CTX:
            hi = N_CTX


        # lo = 0
        # hi = N_CTX
        lo = tl.multiple_of(lo, BLOCK_N)
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
        MASK_block_ptr = tl.advance(MASK_block_ptr, (0, lo))
        if OUTPUT_LOGITS:
            Logits_block_ptr = tl.advance(Logits_block_ptr, (0, lo))
    else:
        lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if IS_EVEN_N:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")

        if IS_EVEN_N and IS_EVEN_M:
            mask = tl.load(MASK_block_ptr)
        else:
            mask = tl.load(MASK_block_ptr, boundary_check=(0,1), padding_option="zero")

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k) 
        qk = qk * qk_scale
        qk = tl.where(mask, qk, float("-inf"))
        if OUTPUT_LOGITS:
            if IS_EVEN_N and IS_EVEN_M:
                tl.store(Logits_block_ptr, qk.to(k.dtype))
            else:
                tl.store(Logits_block_ptr, qk.to(k.dtype), boundary_check=(0,1))
   
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        p = tl.where(mask, p, 0)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        tmp = m_i - m_ij
        alpha_mask = (tmp != tmp) # check nan
        alpha = tl.math.exp2(tmp)
        alpha = tl.where(alpha_mask, 1., alpha)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if IS_EVEN_N:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")

        acc += tl.dot(p.to(v.dtype), v)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        MASK_block_ptr = tl.advance(MASK_block_ptr, (0, BLOCK_N))
        if OUTPUT_LOGITS:
            Logits_block_ptr = tl.advance(Logits_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i


@triton.heuristics(
    {
        "IS_EVEN_M": lambda args: args["N_CTX"] % args["BLOCK_M"] == 0,
        "IS_EVEN_N": lambda args: args["NKV_CTX"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _attn_fwd(Q, K, V, MASK, sm_scale, M, Out, L, Logits,#
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_maskz, stride_maskh, stride_maskm, stride_maskk,
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_logitsz, stride_logitsh, stride_logitsm, stride_logitsn,
              Z, H,  #
              N_CTX,  #
              ROUND_CTX,
              NKV_CTX,
              sliding_window_offset,
              sliding_window_size,
              IS_EVEN_M: tl.constexpr,
              IS_EVEN_N: tl.constexpr,
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              END: tl.constexpr,
              INIT: tl.constexpr,
              SLIDING_WINDOW: tl.constexpr,
              OUTPUT_LOGITS: tl.constexpr
            ):

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    # qvk_offset = off_z * stride_qz + off_h * stride_qh
    mask_offset = off_z.to(tl.int64) * stride_maskz + off_h.to(tl.int64) * stride_maskh
    logits_offset = off_z.to(tl.int64) * stride_logitsz + off_h.to(tl.int64) * stride_logitsh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(NKV_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, NKV_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    MASK_block_ptr = tl.make_block_ptr(
        base=MASK + mask_offset,
        shape = (N_CTX, NKV_CTX),
        strides=(stride_maskm, stride_maskk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(ROUND_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    if OUTPUT_LOGITS:
        Logits_block_ptr = tl.make_block_ptr(
            base = Logits + logits_offset,
            shape = (N_CTX, NKV_CTX),
            strides = (stride_logitsm, stride_logitsn),
            offsets = (start_m * BLOCK_M, 0),
            block_shape = (BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
    else:
        Logits_block_ptr = None
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # initialize pointer to m and l
    m_ptrs = M + off_hz * ROUND_CTX + offs_m
    l_ptrs = L + off_hz * ROUND_CTX + offs_m
    if INIT:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    else:
        # don't have to check boundary for q len 
        m_i = tl.load(m_ptrs).to(tl.float32)
        l_i = tl.load(l_ptrs).to(tl.float32)
        acc = tl.load(O_block_ptr).to(tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634   # 1/log(2)
    # load q: it will stay in SRAM throughout
    if IS_EVEN_M:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, MASK_block_ptr, Logits_block_ptr, #
                                    start_m, qk_scale, NKV_CTX, #
                                    sliding_window_offset, sliding_window_size,
                                    BLOCK_M, BLOCK_DMODEL, BLOCK_N, SLIDING_WINDOW, IS_EVEN_M, IS_EVEN_N, OUTPUT_LOGITS) 
    # epilogue
    if (END):
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
    else:
        tl.store(l_ptrs, l_i)

    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def _forward(q, k, v, mask, sm_scale, o = None, m = None, l = None, end = False, sliding_window=None, output_logits=False):
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    assert mask.dtype in [torch.bool]
    init = False

    BLOCK_M = 64
    BLOCK_N = 64
    q_round_len = math.ceil(q.shape[2] / 128) * 128
    o_shape = (q.shape[0], q.shape[1], q_round_len, q.shape[3])
    m_shape = (q.shape[0], q.shape[1], q_round_len)
    l_shape = (q.shape[0], q.shape[1], q_round_len)

    if o is None:
        o = torch.empty(o_shape, device=q.device, dtype=q.dtype)
        m = torch.empty(m_shape, device=q.device, dtype=torch.float32)
        l = torch.empty(l_shape, device=q.device, dtype=torch.float32)
        init = True

    if output_logits:
        logits = torch.empty(
            mask.shape,
            dtype=q.dtype,
            device=q.device
        )
        logits_strides = (
            logits.stride(0),
            logits.stride(1),
            logits.stride(2),
            logits.stride(3)
        )
    else:
        logits = None
        logits_strides = (0, 0, 0, 0)

    
    if sliding_window is not None:
        sliding_window_offset, sliding_window_size = sliding_window
    else:
        sliding_window_offset, sliding_window_size = None, None

    grid = lambda META: (
        triton.cdiv(q.shape[2], META["BLOCK_M"]),
        q.shape[0] * q.shape[1],
    )
    _attn_fwd[grid](
        q, k, v, mask, sm_scale, m, o, l, logits, #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        *logits_strides,
        q.shape[0], q.shape[1],  #
        q.shape[2],  #
        q_round_len,
        k.shape[2],
        sliding_window_offset,
        sliding_window_size,
        BLOCK_DMODEL=Lk,  #
        END=end,
        INIT=init,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SLIDING_WINDOW=(sliding_window is not None),
        OUTPUT_LOGITS=output_logits,
        num_warps=4,
        num_stages=4
    )

    if end:
        o = o[:, :, :q.shape[2], :].contiguous().to(q.dtype)

    return o, m, l, logits


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, q1, k1, v1, mask1, q2, k2, v2, mask2, sm_scale = None, 
        sliding_window1=None, sliding_window2=None, 
        output_M: bool = False, output_logits1=False, output_logits2=False
    ):

        q1 = q1.contiguous()
        k1 = k1.contiguous()
        v1 = v1.contiguous()
        mask1 = mask1.contiguous()
        q2 = q2.contiguous()
        k2 = k2.contiguous()
        v2 = v2.contiguous()
        mask2 = mask2.contiguous()

        m1_shape = mask1.shape
        m2_shape = mask2.shape
        assert len(m1_shape) <= 4
        assert len(m2_shape) <= 4
        m1_shape = [1] * (4-len(m1_shape)) + list(m1_shape)
        m2_shape = [1] * (4-len(m2_shape)) + list(m2_shape)
        mask1 = mask1.view(m1_shape)
        mask2 = mask2.view(m2_shape)

        assert q1.shape[-2] == q2.shape[-2]
        if sm_scale is None:
            import math
            sm_scale = 1 / math.sqrt(q1.shape[-1])

        BATCH, N_HEAD, N_CTX = q1.shape[:3]
        KV1_CTX = k1.shape[-2]
        KV2_CTX = k2.shape[-2]
        mask1 = mask1.expand((BATCH, N_HEAD, N_CTX, KV1_CTX))
        mask2 = mask2.expand((BATCH, N_HEAD, N_CTX, KV2_CTX))
        o, m, l, logits1 = _forward(q1, k1, v1, mask1, sm_scale, sliding_window=sliding_window1, output_logits=output_logits1)
        o, m, l, logits2 = _forward(q2, k2, v2, mask2, sm_scale, o, m, l, end=True, sliding_window=sliding_window2, output_logits=output_logits2)

        # ctx.save_for_backward(q1, k1, v1, mask1, q2, k2, v2, mask2, o, m)
        # ctx.sm_scale = sm_scale

        if output_M:
            return o, m[:, :, :N_CTX], logits1, logits2

        return o

    @staticmethod
    def backward(ctx, do, *args, **kwargs):

        raise NotImplementedError


from typing import Optional, Tuple, Union

def mq_attn_triton(
    q1: torch.FloatTensor,
    k1: torch.FloatTensor,
    v1: torch.FloatTensor,
    mask1: torch.BoolTensor,
    q2: torch.FloatTensor,
    k2: torch.FloatTensor,
    v2: torch.FloatTensor,
    mask2: torch.BoolTensor,
    sm_scale: Optional[float] = None,
    sliding_window1: Optional[Union[Tuple[int, int], int]] = None, 
    sliding_window2: Optional[Union[Tuple[int, int], int]] = None,
    casual1: Optional[bool] = False,
    casual2: Optional[bool] = False,
    output_M: bool = False,
    output_logits1: bool = False,
    output_logits2: bool = False
) -> torch.FloatTensor:
    if casual1:
        assert sliding_window1 is None
        sliding_window1 = (k1.shape[2] - q1.shape[2], k1.shape[2])
    if casual2:
        assert sliding_window2 is None
        sliding_window2 = (k2.shape[2] - q2.shape[2], k2.shape[2])

    if isinstance(sliding_window1, int):
        sliding_window1 = (
            k1.shape[2] - q1.shape[2], sliding_window1
        )

    if isinstance(sliding_window2, int):
        sliding_window2 = (
            k2.shape[2] - q2.shape[2], sliding_window2
        )

    return _attention.apply(
        q1, k1, v1, mask1, q2, k2, v2, mask2, sm_scale, sliding_window1, sliding_window2,
        output_M, output_logits1, output_logits2
    )

if __name__ == "__main__":
    Q_LEN = 512
    KV1_LEN = 512
    KV2_LEN = 0
    DIM = 128
    HD = 32
    BT = 1

    sliding_window_size1 = 8192
    sliding_window_size2 = 8192
    casual1 = False
    casual2 = False

    if casual1:
        assert sliding_window_size1 is None
        sliding_window_size1 = KV1_LEN
    if casual2:
        assert sliding_window_size2 is None
        sliding_window_size2 = KV2_LEN

    dtype = torch.bfloat16
    q1 = torch.randn((BT, HD, Q_LEN, DIM), dtype=dtype, device='cuda', requires_grad=True)
    q2 = torch.randn((BT, HD, Q_LEN, DIM), dtype=dtype, device='cuda', requires_grad=True)

    k1 = torch.randn((BT, HD, KV1_LEN, DIM), dtype=dtype, device='cuda', requires_grad=True)
    k2 = torch.randn((BT, HD, KV2_LEN, DIM), dtype=dtype, device='cuda', requires_grad=True)

    v1 = torch.randn((BT, HD, KV1_LEN, DIM), dtype=dtype, device='cuda', requires_grad=True) 
    v2 = torch.randn((BT, HD, KV2_LEN, DIM), dtype=dtype, device='cuda', requires_grad=True)

    mask1 = torch.randint(0, 2, (BT, HD, Q_LEN, KV1_LEN), dtype=torch.bool, device='cuda')
    mask1 = mask1 | (~mask1)
    if sliding_window_size1 is not None:
        dist = torch.arange(0, Q_LEN, dtype=torch.int64, device='cuda')[:, None] - torch.arange(0, KV1_LEN, dtype=torch.int64, device='cuda')[None, :] + (KV1_LEN - Q_LEN)
        mask1 &= ((dist >= 0) & (dist < sliding_window_size1))[None, None, :, :]

    mask2 = torch.randint(0, 2, (BT, HD, Q_LEN, KV2_LEN), dtype=torch.bool, device='cuda')
    mask2 = mask2 | (~mask2)
    if sliding_window_size2 is not None:
        dist = torch.arange(0, Q_LEN, dtype=torch.int64, device='cuda')[:, None] - torch.arange(0, KV2_LEN, dtype=torch.int64, device='cuda')[None, :] + (KV2_LEN - Q_LEN)
        mask2 &= ((dist >= 0) & (dist < sliding_window_size2))[None, None, :, :]


    def do_bench(*, fn, warmup = 5, repeat = 5, **kwargs):
        for _ in range(warmup):
            ret = fn(**kwargs)
        tl = 0
        for _ in range(repeat):
            _st = time.time()
            torch.cuda.synchronize()
            ret = fn(**kwargs)
            torch.cuda.synchronize()
            tl += time.time() - _st
        return ret, tl / repeat


    import time
    fwd_args = {
        "q1": q1,
        "k1": k1,
        "v1": v1,
        "mask1": mask1,
        "q2": q2,
        "k2": k2,
        "v2": v2,
        "mask2": mask2,
    }
    if sliding_window_size1 is not None:
        sliding_window1 = sliding_window_size1
    else:
        sliding_window1 = None

    if sliding_window_size2 is not None:
        sliding_window2 = sliding_window_size2
    else:
        sliding_window2 = None


    ret, tm = do_bench(fn = mq_attn_triton, sliding_window1=sliding_window1, sliding_window2=sliding_window2, **fwd_args)
    print("fattn:")
    print(" - fwd:", tm)
    if sliding_window2 is not None or sliding_window1 is not None:
        ret2, tm = do_bench(fn = mq_attn_triton, **fwd_args)
        print(" - no sliding window fwd:", tm)
        print(" - sliding window error: ", (ret- ret2).abs().max().item())


    from mq_attn_torch import mq_attn_torch
    torch_impl = mq_attn_torch
    tret, tm = do_bench(fn=torch_impl, **fwd_args)
    print("")
    print("torch:")
    print(" - fwd:", tm)

    print(((ret - tret).abs()).max())