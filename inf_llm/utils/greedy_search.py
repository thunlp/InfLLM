import torch
import numpy as np

class GreedySearch:
    def __init__(self, model, tokenizer):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.past_kv = None

    def clear(self):
        self.past_kv = None

    def _process_texts(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0).cuda()

        return model_inputs


    def generate(self, text=None, input_ids=None, top_k=20, top_p=0.9, temperature=0.95, do_sample=True, repetition_penalty=1.05, **kwargs):
        if input_ids is None:
            model_inputs = self._process_texts(text)
            input_ids = model_inputs['input_ids']

        with torch.inference_mode():
            result = self._decode(input_ids, top_k=top_k, top_p=top_p, temperature=temperature, do_sample=do_sample, repetition_penalty=repetition_penalty, **kwargs)
        return result

    def _decode(self, input_ids, max_length=100, extra_end_token_ids=[], chunk_size: int = 4096, output=False, top_k=20, top_p=0.9, temperature=0.95, do_sample=True, repetition_penalty=1.05):
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.cuda()
        attention_mask = torch.ones_like(input_ids)
        assert input_ids.size(0) == 1
        length = input_ids.size(1)
        end_token_ids = extra_end_token_ids + [self.tokenizer.eos_token_id]
        logits = None
        past_key_values = self.past_kv
        if output:
            output_text = ""
        
        for i in range(max_length + 1):
            if i == 0:
                if chunk_size is None:
                    chunk_size = input_ids.size(1)
                for st in range(0, input_ids.size(1) - 1, chunk_size):
                    ed = min(input_ids.size(1) - 1, st + chunk_size)
                    out = self.model(
                        input_ids = input_ids[:, st: ed],
                        attention_mask = attention_mask[:, :ed],
                        use_cache = True,
                        return_dict = True,
                        past_key_values = past_key_values
                    )
                    logits, past_key_values = out.logits, out.past_key_values

                out = self.model(
                    input_ids = input_ids[:, -1:],
                    attention_mask = attention_mask,
                    use_cache = True,
                    return_dict = True,
                    past_key_values = past_key_values
                )
                logits, past_key_values = out.logits, out.past_key_values
            else:
                out = self.model(
                    input_ids = input_ids[:, -1:],
                    attention_mask = attention_mask,
                    past_key_values = past_key_values,
                    use_cache = True,
                    return_dict = True
                )
                logits, past_key_values = out.logits, out.past_key_values

            logits = logits[:, -1, :] / temperature  # Applying temperature
            
            if do_sample:
                # Apply sampling
                if top_k > 0:
                    # Apply top-k sampling
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    sorted_logits[sorted_indices_to_remove] = -float('Inf')
                    sorted_logits = sorted_logits[:, :top_k]
                    sorted_indices = sorted_indices[:, :top_k]
                    probs = torch.softmax(sorted_logits, dim=-1)
                    chosen_index = torch.multinomial(probs, 1)
                    word = sorted_indices[0][chosen_index[0].item()].unsqueeze(0)
                else:
                    # Apply nucleus sampling (top-p sampling)
                    probs = torch.softmax(logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    indices_to_remove = cumulative_probs > top_p
                    indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
                    indices_to_remove[..., 0] = 0
                    probs[indices_to_remove] = 0
                    chosen_index = torch.multinomial(probs, 1)
                    word = sorted_indices[0][chosen_index[0].item()].unsqueeze(0)
            else:
                # Standard greedy decoding
                word = logits.argmax(dim=-1)
                
            if word.item() in end_token_ids or i == max_length:
                break

            if repetition_penalty != 1.0:
                # Apply repetition penalty
                if input_ids[0].tolist().count(word.item()) > 1:
                    logits[:, word.item()] /= repetition_penalty

            input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.int, device=attention_mask.device)),
                dim=-1
            )
            if output:
                tmp = self.tokenizer.decode(input_ids.squeeze(0)[length:])
                if len(tmp) > len(output_text):
                    import sys               
                    sys.stdout.write(tmp[len(output_text):])
                    sys.stdout.flush()
                    output_text = tmp

        self.past_kv = past_key_values

        if output:
            sys.stdout.write("\n")
            sys.stdout.flush()

        return [self.tokenizer.decode(input_ids.squeeze(0)[length:])]
