import math
import torch

def mq_attn_torch(
    q1: torch.FloatTensor, 
    k1: torch.FloatTensor, 
    v1: torch.FloatTensor, 
    mask1: torch.BoolTensor, 
    q2: torch.FloatTensor, 
    k2: torch.FloatTensor, 
    v2: torch.FloatTensor, 
    mask2: torch.BoolTensor, 
    sm_scale=None,
    return_score: bool = False,
    **kwargs
):
    assert q1.shape == q2.shape
    K1_LEN = k1.shape[-2]

    m1_shape = mask1.shape
    m2_shape = mask2.shape
    m1_shape = [1] * (4-len(m1_shape)) + list(m1_shape)
    m2_shape = [1] * (4-len(m2_shape)) + list(m2_shape)
    mask1 = mask1.view(m1_shape)
    mask2 = mask2.view(m2_shape)

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(q1.shape[-1])

    s1 = torch.matmul(q1.float(), k1.transpose(-1, -2).float())
    s1 = torch.masked_fill(
        s1,
        mask1 == False,
        float("-inf")
    )

    s2 = torch.matmul(q2.float(), k2.transpose(-1, -2).float())
    s2 = torch.masked_fill(
        s2,
        mask2 == False,
        float("-inf")
    )

    s = torch.cat((s1, s2), dim=-1) * sm_scale
    p = torch.softmax(s, dim=-1)
    p1 = p[:, :, :, :K1_LEN].contiguous()
    p2 = p[:, :, :, K1_LEN:].contiguous()

    p1 = torch.masked_fill(
        p1,
        mask1==False,
        0.
    )

    p2 = torch.masked_fill(
        p2,
        mask2==False,
        0.
    )

    o = torch.matmul(p1, v1.float()) + torch.matmul(p2, v2.float())

    if return_score:
        return o.to(q1.dtype), p1, p2

    return o.to(q1.dtype)
