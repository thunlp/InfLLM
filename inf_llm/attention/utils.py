import torch

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def get_mq_attn(flash_attn=False):

    fattn = False

    class UseTorch(Exception):
        pass

    try:
        if flash_attn:
            from ..mq_attn_triton import mq_attn_triton as mq_attn
            fattn = True
        else:
            raise UseTorch

    except Exception as E:
        if not isinstance(E, UseTorch):
            print("Load Triton Flash Attention Error.")

        from ..mq_attn_torch import mq_attn_torch as mq_attn

    return mq_attn, fattn

