from .base import MultiStageDotProductionAttention
from typing import Tuple

def get_multi_stage_dot_production_attention(flash_attn=False) -> Tuple[type, bool]:
    class UseTorch(Exception):
        pass

    try:
        if flash_attn:
            from .triton_impl import TritonMultiStageDotProductionAttention as ret
            fattn = True
        else:
            raise UseTorch

    except Exception as E:
        fattn = False
        if not isinstance(E, UseTorch):
            if get_multi_stage_dot_production_attention.warn:
                from warnings import warn
                warn("Load triton flash attention error. Use torch impl.")
                get_multi_stage_dot_production_attention.warn = False

        from .torch_impl import TorchMultiStageDotProductionAttention as ret


    return ret, fattn


get_multi_stage_dot_production_attention.warn = True
