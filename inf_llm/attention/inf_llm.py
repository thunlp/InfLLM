import torch
from typing import Optional
from .context_manager import ContextManager

def inf_llm_forward(
    n_local, n_init, topk, 
    block_size, max_cached_block,
    exc_block_size, score_decay, fattn,
    perhead=False,
    repr_topk: int = 1,
    max_calc_block = None,
    use_buffer=True,
    cache_strategy="lru",
    calc_block_score=False,
    ignore_remainder=False,
    chunk_topk_calc=None,
    async_global_stream=False,
    *args, **kwargs
):

    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : Optional[torch.Tensor],
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out, 
                    dim_head, num_heads, num_heads_kv
    ):

        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)


        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)


        if past_key_value is None:
            past_key_value = ContextManager(
                position_bias, n_init,
                n_local, block_size,
                max_cached_block, topk,
                exc_block_size, perhead,
                score_decay, fattn, repr_topk,
                max_calc_block, use_buffer,
                cache_strategy, calc_block_score,
                ignore_remainder, chunk_topk_calc,
                async_global_stream
            )


        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        o = past_key_value.append(
            local_q, local_k, local_v,
            global_q, global_k, global_v,
        )


        o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
        o = o.reshape(batch_size, len_q, dim_head * num_heads)
        o = attention_out(o)

        if use_cache:
            return o, past_key_value
        else:
            return o


    return forward
