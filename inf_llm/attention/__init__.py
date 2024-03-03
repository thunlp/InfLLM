from .rope import RotaryEmbeddingESM

from .inf_llm import inf_llm_forward
from .infinite_lm import infinite_lm_forward
from .stream_llm import stream_llm_forward
from .origin import origin_forward

ATTN_FORWRAD = {
    "inf-llm": inf_llm_forward,
    "infinite-lm": infinite_lm_forward,
    "stream-llm": stream_llm_forward,
    "origin": origin_forward
}

__all__ = ["RotaryEmbeddingESM", "ATTN_FORWARD"]