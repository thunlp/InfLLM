import torch
from typing import Optional

def origin_forward(fattn: bool, *args, **kwargs):
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


        if past_key_value is not None:
            h_k = torch.cat((past_key_value[0], h_k), dim=-2)
            h_v = torch.cat((past_key_value[1], h_v), dim=-2)
            len_k = h_k.size(-2)

        if use_cache:
            current_key_value = h_k, h_v

        h_q, h_k = position_bias(h_q, h_k)

        if fattn:
            from flash_attn.flash_attn_interface import flash_attn_func
            h_q = h_q.transpose(1, 2)
            h_k = h_k.transpose(1, 2)
            h_v = h_v.transpose(1, 2)
            o = flash_attn_func(h_q, h_k, h_v, causal=True)
        else:
            dist = torch.arange(0, len_q, device=h_q.device)[:, None] - torch.arange(0, len_k, device=h_q.device)[None, :] + len_k - len_q
            attention_mask = (dist >= 0)
            score = torch.matmul(h_q, h_k.transpose(-1, -2))
            score = torch.masked_fill(
                score,
                attention_mask.view(1, 1, len_q, len_k)==False,
                torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype)
            )   # (batch, num_heads, len_q, len_k)

            score = torch.nn.functional.softmax(score, dim=-1)

            # avoid nan in softmax
            score = torch.masked_fill(
                score,
                attention_mask.view(1, 1, len_q, len_k)==False,
                torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
            )
            #.view(batch_size * self.num_heads, len_q, len_k) # (batch * num_heads, len_q, len_k)


            # (batch * num_heads, len_q, len_k) @ (batch * num_heads, len_k, dim_head) = (batch * num_heads, len_q, dim_head)
            o = torch.matmul(score, h_v)

            o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)

        o = o.reshape(batch_size, len_q, dim_head * num_heads)
        o = attention_out(o)

        if use_cache:
            return o, current_key_value
        else:
            return o

    return forward