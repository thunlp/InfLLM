import torch

class MultiStageDotProductionAttention:
    def __init__(
        self, 
        q_shape,
        dtype,
        device,
    ):
        self.q_shape = q_shape
        self.dtype = dtype
        self.device = device
        self.end = False
        self.ret = torch.zeros(
            q_shape, dtype=dtype, device=device
        )
        self.score_list = []

    def append(
        self, 
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, 
        end=False, get_score=False,
        *args, **kwargs
    ):
        raise NotImplementedError


    def get_result(self):
        return self.ret, self.score_list
