import torch
from ..attention import RotaryEmbeddingESM, ATTN_FORWRAD

def huggingface_forward(forward):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        assert not output_attentions
        ret = forward(
            self, hidden_states, hidden_states,
            position_ids, use_cache, past_key_value,
            self.q_proj, self.k_proj, self.v_proj, self.o_proj, 
            self.head_dim, self.num_heads, self.num_key_value_heads
        )
        if use_cache:
            o, pkv = ret
        else:
            o = ret
            pkv = None

        return o, None, pkv

    return hf_forward


def patch_hf(
    model,
    attn_type: str = "inf_llm",
    attn_kwargs: dict = {},
    base = None, 
    distance_scale = None,
    **kwargs
):
    attn_kwargs.update(kwargs)
    # This approach lacks scalability and will be refactored.
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel, BaseModelOutputWithPast
    from transformers.models.mistral.modeling_mistral import MistralAttention, MistralModel
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Model

    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        *args,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb

        if use_cache:
            pkv = tuple()

        else:
            pkv = None
            

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=self.position_bias,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                _cache = layer_outputs[2 if output_attentions else 1]
                pkv = pkv + (_cache,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, pkv, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    forward = huggingface_forward(ATTN_FORWRAD[attn_type](**attn_kwargs))

    if isinstance(model, LlamaForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif isinstance(model, MistralForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif isinstance(model, Qwen2ForCausalLM):
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    elif model.__class__.__name__ == "MiniCPMForCausalLM":
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    else:
        raise ValueError("Only supports llama, mistral and qwen2 models.")

    hf_rope = model.model.layers[0].self_attn.rotary_emb 
    base = base if base is not None else hf_rope.base
    distance_scale = distance_scale if distance_scale is not None else 1.0
    rope = RotaryEmbeddingESM(
        hf_rope.dim,
        base,
        distance_scale
    )
    model.model.position_bias = rope

    def set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)

    model.apply(set_forward)

    model.model._old_forward = model.model.forward
    model.model.forward = model_forward.__get__(model.model, Model)

    return model

