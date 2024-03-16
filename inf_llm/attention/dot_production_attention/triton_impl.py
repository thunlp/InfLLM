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
from .base import MultiStageDotProductionAttention


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


def _forward(q, k, v, mask, sm_scale, o = None, m = None, l = None, end = False, sliding_window=None, output_logits=False, init=False):
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    assert mask.dtype in [torch.bool]

    BLOCK_M = 64
    BLOCK_N = 64
    q_round_len = math.ceil(q.shape[2] / 128) * 128

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



class TritonMultiStageDotProductionAttention(MultiStageDotProductionAttention):
    def __init__(self, q_shape, dtype, device):
        self.q_shape = q_shape
        self.dtype = dtype
        self.device = device
        q_round_len = math.ceil(q_shape[2] / 128) * 128
        o_shape = (q_shape[0], q_shape[1], q_round_len, q_shape[3])
        m_shape = (q_shape[0], q_shape[1], q_round_len)
        l_shape = (q_shape[0], q_shape[1], q_round_len)

        self.o = torch.empty(o_shape, device=device, dtype=torch.float32)
        self.m = torch.empty(m_shape, device=device, dtype=torch.float32)
        self.l = torch.empty(l_shape, device=device, dtype=torch.float32)
        self.mask_list = []
        self.logits_list = []
        self.score_list = []
        self.end = False
        self.init = False

    def finalize(self):
        self.end = True
        for logits, mask in zip(self.logits_list, self.mask_list):
            if logits is not None:
                if self.m.size(-1) != self.q_shape[-2]:
                    self.m = self.m[:, :, :self.q_shape[-2]]
                logits.sub_(self.m[:, :, :, None])
                logits.exp2_()
                logits.masked_fill_(
                    mask == False,
                    0
                )
                self.score_list.append(logits)
            else:
                self.score_list.append(None)

        self.ret = self.o



    def append(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, end=False, get_score=False, casual = False, sliding_window = None):
        assert q.shape == self.q_shape
        if casual:
            assert sliding_window is None
            sliding_window = (k.shape[2] - q.shape[2], k.shape[2])

        if isinstance(sliding_window, int):
            sliding_window = (
                k.shape[2] - q.shape[2], sliding_window
            )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        mask = mask.contiguous()
        mask_shape = [1] * (4-mask.dim()) + list(mask.shape)
        mask = mask.view(mask_shape)
        
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        KV_CTX = k.shape[-2]
        mask = mask.expand((BATCH, N_HEAD, N_CTX, KV_CTX))
        sm_scale = 1 / math.sqrt(q.shape[-1])
        o, m, l, logits = _forward(q, k, v, mask, sm_scale, self.o, self.m, self.l, sliding_window=sliding_window, output_logits=get_score, end=end, init=not self.init)
        self.init = True
        self.o = o
        self.m = m
        self.l = l
        if get_score:
            self.mask_list.append(mask)
            self.logits_list.append(logits)
        else:
            self.mask_list.append(None)
            self.logits_list.append(None)

        if end:
            assert not self.end 
            self.finalize()

