import math
import torch

from .base import MultiStageDotProductionAttention

class TorchMultiStageDotProductionAttention(MultiStageDotProductionAttention):
    def __init__(self, q_shape, dtype, device):
        super().__init__(q_shape, dtype, device)
        self.logits_list = []
        self.v_list = []
        self.mask_list = []
        self.get_score_list = []
        self.kv_len_list = []

    def finalize(self):
        logits = torch.cat(self.logits_list, dim=-1)
        p = torch.softmax(logits, dim=-1)
        st = 0
        for kv_len, mask, get_score, v in zip(self.kv_len_list, self.mask_list, self.get_score_list, self.v_list):
            ed = st + kv_len
            tmp = p[:, :, :, st: ed]
            tmp = torch.masked_fill(
                tmp,
                mask==False,
                0
            )
            if get_score:
                self.score_list.append(tmp)
            else:
                self.score_list.append(None)

            self.ret.add_(
                torch.matmul(tmp, v)
            )

            st = ed


    def append(
            self, 
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, 
            end=False, get_score=False,
            *args, **kwargs
        ):
        m_shape = [1] * (4-mask.dim()) + list(mask.shape)
        mask = mask.view(m_shape)
        self.v_list.append(v)
        self.mask_list.append(mask)
        self.get_score_list.append(get_score)
        self.kv_len_list.append(k.size(-2))
        logits = torch.matmul(q, k.transpose(-1, -2))
        logits = torch.masked_fill(
            logits,
            mask==False,
            float("-inf")
        )
        logits.mul_(1/math.sqrt(q.size(-1)))
        self.logits_list.append(logits)

        if end:
            self.finalize()
