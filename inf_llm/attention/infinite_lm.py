import torch
from .utils import repeat_kv
from .dot_production_attention import get_multi_stage_dot_production_attention


def infinite_lm_forward(n_local, n_init, fattn: bool = False, *args, **kwargs):
    Attn, _ = get_multi_stage_dot_production_attention(fattn)
    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : torch.Tensor,
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out,
                    dim_head, num_heads, num_heads_kv
    ):

        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        
        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3)   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3)   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3)   # (batch, num_heads_kv, len_k, dim_head)

        h_q = h_q.contiguous()      # (batch * num_heads, len_q, dim_head)
        h_k = h_k.contiguous()      # (batch * num_heads, len_k, dim_head)
        h_v = h_v.contiguous()      # (batch * num_heads, len_k, dim_head)

        if past_key_value is not None:
            h_k = torch.cat([past_key_value[0], h_k], dim=-2)
            h_v = torch.cat([past_key_value[1], h_v], dim=-2)

            len_k += past_key_value[2]

        if use_cache:
            if len_k <= n_local + n_init:
                h_k_cache = h_k
                h_v_cache = h_v
            else:
                h_k_cache = torch.cat([h_k[:,:, :n_init, :], h_k[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
                h_v_cache = torch.cat([h_v[:,:, :n_init, :], h_v[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)


            current_key_value = (h_k_cache, h_v_cache, len_k)

        else:
            current_key_value = None

        h_q_, h_k_, h_v_ = h_q, h_k, h_v
        if len_q + n_local < h_k_.size(-2):
            h_k_ = h_k_[:, :, h_k_.size(-2) - len_q - n_local:, :]
            h_v_ = h_v_[:, :, h_v_.size(-2) - len_q - n_local:, :]


        if num_heads != num_heads_kv:
            assert num_heads % num_heads_kv == 0
            num_group = num_heads // num_heads_kv
            h_k_ = repeat_kv(h_k_, num_group)
            h_v_ = repeat_kv(h_v_, num_group)

        local_h_q, local_h_k = position_bias(h_q_, h_k_)
        local_h_v = h_v_

        dist = torch.arange(len_q, device=h_q.device).unsqueeze(1) - torch.arange(local_h_k.size(2), device=h_q.device).unsqueeze(0) + (local_h_k.size(2) - len_q)
        local_mask = (dist >= 0) & (dist < n_local)

        if len_k > n_local:
            init_h_q = position_bias.apply_rotary_pos_emb(
                h_q,
                1, n_local, position_bias._cos_cached, position_bias._sin_cached
            )
            dist = torch.arange(len_q, device=h_q.device).unsqueeze(1) - torch.arange(n_init, device=h_q.device).unsqueeze(0) + (len_k - len_q)
            init_mask = (dist >= n_local)[:, :n_init].contiguous()
            init_h_k = h_k
            init_h_v = h_v
            init_h_k = init_h_k[:, :, :n_init, :].contiguous()
            init_h_v = init_h_v[:, :, :n_init, :].contiguous()
            if num_heads != num_heads_kv:
                assert num_heads % num_heads_kv == 0
                num_group = num_heads // num_heads_kv
                init_h_k = repeat_kv(init_h_k, num_group)
                init_h_v = repeat_kv(init_h_v, num_group)

        else:
            init_h_q = h_q
            init_mask = torch.empty(
                (batch_size, num_heads, len_q, 0),
                dtype=torch.bool,
                device=h_q.device
            )
            init_h_k = torch.empty(
                (batch_size, num_heads, 0, dim_head),
                device=h_k.device,
                dtype=h_k.dtype
            )
            init_h_v = torch.empty(
                (batch_size, num_heads, 0, dim_head),
                device=h_v.device,
                dtype=h_v.dtype
            )

        attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(local_h_q, local_h_k, local_h_v, local_mask, sliding_window=n_local)
        attn.append(init_h_q, init_h_k, init_h_v, init_mask, end=True)
        score, _ = attn.get_result()

        score = score.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3) # (batch, len_q, num_heads, dim_head)
        score = score.reshape(batch_size, len_q, num_heads * dim_head) # (batch, len_q, num_heads * dim_head)

        score = attention_out(score)

        if use_cache:
            return score, current_key_value
        else:
            return score

    return forward
