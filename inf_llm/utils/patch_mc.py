import torch
from ..attention import RotaryEmbeddingESM, ATTN_FORWRAD

def model_center_forward(forward):
    def mc_forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    attention_mask : torch.Tensor, # use less
                    position_bias  = None,
                    use_cache: bool = False,
                    past_key_value = None,
    ):
        return forward(
            self, query, key_value,
            position_bias, use_cache, past_key_value,
            self.project_q, self.project_k, self.project_v, self.attention_out,
            self.dim_head, self.num_heads, self.num_heads_kv
        )

    return mc_forward

def patch_model_center(
    model,
    attn_type: str = "inf-llm",
    attn_kwargs: dict = {},
    base = None,
    distance_scale = None,
    **kwargs
):
    attn_kwargs.update(kwargs)
    from model_center.model import Llama
    from model_center.layer import Attention
    from model_center.model import BaseModelOutput
    from bmtrain.wrapper import make_distributed
    from typing import Optional, List

    def model_forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                length: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = False,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                output_logits: Optional[bool] = True,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                return_dict: Optional[bool] = True,
        ): # https://github.com/OpenBMB/ModelCenter/blob/main/model_center/model/llama.py
        if inputs_embeds is None:
            hidden_states = self.input_embedding(input_ids)
        else:
            hidden_states = inputs_embeds
            
        # input the input embeddings into the LLaMa model
        current_key_values = None
        if use_cache:
            hidden_states, current_key_values = self.encoder(hidden_states, attention_mask, self.position_bias, 
                                                             use_cache = use_cache, past_key_values = past_key_values)
        else:
            hidden_states = self.encoder(hidden_states, attention_mask, self.position_bias)

        
        # use the hidden states of the last layer for sequential tasks, such as sequential labeling and language modeling.
        logits = None
        if output_logits:
            if self.config.cls_head:
                logits = self.cls_projection(hidden_states)
            elif self.config.tied:
                logits = self.input_embedding.projection(hidden_states)
            elif not self.config.tied:
                logits = self.output_projection(hidden_states)

        # BaseModelOutput or tuple: The LLaMa output. 
        if not return_dict:
            return hidden_states, current_key_values, logits, None, None
        else:
            return BaseModelOutput(
                last_hidden_state = hidden_states,
                past_key_values = current_key_values,
                logits = logits,
                hidden_states = None,
                attentions = None,
            )


    forward = model_center_forward(ATTN_FORWRAD[attn_type](**attn_kwargs))

    assert isinstance(model, Llama)
    config = model.config
    rope = RotaryEmbeddingESM(
        config.dim_head, 
        base if base is not None else model.position_bias.base, 
        distance_scale if distance_scale is not None else model.position_bias.distance_scale, 
    )
    model.position_bias = make_distributed(rope)

    def set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)

    model.apply(set_forward)

    model._old_forward = model.forward
    model.forward = model_forward.__get__(model, Llama)

    return model

