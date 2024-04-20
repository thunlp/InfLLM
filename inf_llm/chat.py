"""
This code is based on modifications to the FastChat code originally developed by the LMSYS team. 
The original project is licensed under the Apache License 2.0. 

## Changes Made

 - This code incorporates support for the InfLLM patch.

"""


"""Inference for FastChat models."""
import abc
import gc
import json
import math
import os
import sys
import time
from typing import Iterable, Optional, Dict
import warnings
import torch
import argparse

from fastchat.serve.inference import (
    ChatIO, GptqConfig, AWQConfig, 
    ExllamaConfig, XftConfig, 
    load_model, 
    get_context_length,
    get_conv_template,
    get_conversation_template,
    is_partial_stop,
    is_sentence_complete,
    prepare_logits_processor
)

from fastchat.serve.cli import (
    SimpleChatIO, 
    RichChatIO,
    ProgrammaticChatIO,
    str_to_torch_dtype,
    add_model_args
)

from inf_llm.utils import patch_hf

@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
    clear_kv_cache: bool = True,
):
    if hasattr(model, "device"):
        device = model.device

    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    logprobs = params.get("logprobs", None)  # FIXME: Support logprobs>1.
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    
    out = None
    if clear_kv_cache:
        past_key_values = None
    else:
        if hasattr(model, "_fschat_pkv"):
            past_key_values = model._fschat_pkv
        else:
            past_key_values = None
        
    def get_length(pkv):
        if pkv is None:
            return 0

        pkv = pkv[0]
        if isinstance(pkv, tuple):
            if len(pkv) == 3:
                return pkv[-1]
            else:
                return pkv[0].size(-2)

        return pkv.length

    start_length = get_length(past_key_values)

    assert len(input_ids) > start_length
    input_ids = input_ids[start_length:]
    if model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        start_ids = torch.as_tensor([input_ids], device=device)

    token_logprobs = [None]  # The first token has no logprobs.
    sent_interrupt = False
    finish_reason = None
    stopped = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            out = model(input_ids=start_ids, use_cache=True, past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

            if logprobs is not None:
                # Prefull logprobs for the prompt.
                shift_input_ids = start_ids[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                for label_id, logit in zip(
                    shift_input_ids[0].tolist(), shift_logits[0]
                ):
                    token_logprobs.append(logit[label_id])
        else:  # decoding
            out = model(
                input_ids=torch.as_tensor(
                    [[token] if not sent_interrupt else output_ids],
                    device=device,
                ),
                use_cache=True,
                past_key_values=past_key_values if not sent_interrupt else None,
            )
            sent_interrupt = False
            logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)
        if logprobs is not None:
            # Cannot use last_token_logits because logprobs is based on raw logits.
            token_logprobs.append(
                torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
            )

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            ret_logprobs = None
            if logprobs is not None:
                ret_logprobs = {
                    "text_offset": [],
                    "tokens": [
                        tokenizer.decode(token)
                        for token in (
                            output_ids if echo else output_ids[input_echo_len:]
                        )
                    ],
                    "token_logprobs": token_logprobs
                    if echo
                    else token_logprobs[input_echo_len:],
                    "top_logprobs": [{}]
                    * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                }
                # Compute text_offset
                curr_pos = 0
                for text in ret_logprobs["tokens"]:
                    ret_logprobs["text_offset"].append(curr_pos)
                    curr_pos += len(text)

            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "logprobs": ret_logprobs,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    else:
        finish_reason = "length"

    if stopped:
        finish_reason = "stop"

    yield {
        "text": output,
        "logprobs": ret_logprobs,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    model._fschat_pkv = past_key_values

    del out
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()


def chat_loop(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    dtype: Optional[torch.dtype],
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    conv_system_msg: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chatio: ChatIO,
    gptq_config: Optional[GptqConfig] = None,
    awq_config: Optional[AWQConfig] = None,
    exllama_config: Optional[ExllamaConfig] = None,
    xft_config: Optional[XftConfig] = None,
    inf_llm_config: Optional[dict] = None,
    revision: str = "main",
    judge_sent_end: bool = True,
    debug: bool = True,
    history: bool = True,
    clear_kv_cache = False,
    top_k: int = -1,
    top_p: float = 1.0
):
    # Model
    model, tokenizer = load_model(
        model_path,
        device=device,
        num_gpus=num_gpus,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=load_8bit,
        cpu_offloading=cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        exllama_config=exllama_config,
        xft_config=xft_config,
        revision=revision,
        debug=debug,
    )
    if inf_llm_config is not None:
        model = patch_hf(model, inf_llm_config.type,  **inf_llm_config)
    generate_stream_func = generate_stream

    model_type = str(type(model)).lower()
    is_t5 = "t5" in model_type
    is_codet5p = "codet5p" in model_type
    is_xft = "xft" in model_type

    # Hardcode T5's default repetition penalty to be 1.2
    if is_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2

    # Set context length
    context_len = get_context_length(model.config)

    if inf_llm_config is not None:
        context_len = 2147483647

    # Chat
    def new_chat():
        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        if conv_system_msg is not None:
            conv.set_system_message(conv_system_msg)
        return conv

    def reload_conv(conv):
        """
        Reprints the conversation from the start.
        """
        for message in conv.messages[conv.offset :]:
            chatio.prompt_for_output(message[0])
            chatio.print_output(message[1])

    conv = None

    if not history:
        clear_kv_cache = True

    _clear_kv_cache = True
    while True:
        if not history or not conv:
            conv = new_chat()

        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""

        if inp == "!!exit" or not inp:
            print("exit...")
            break
        elif inp == "!!reset":
            print("resetting...")
            _clear_kv_cache = True
            conv = new_chat()
            continue
        elif inp == "!!remove":
            print("removing last message...")
            if len(conv.messages) > conv.offset:
                _clear_kv_cache = True
                # Assistant
                if conv.messages[-1][0] == conv.roles[1]:
                    conv.messages.pop()
                # User
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()
                reload_conv(conv)
            else:
                print("No messages to remove.")
            continue
        elif inp == "!!regen":
            print("regenerating last message...")
            if len(conv.messages) > conv.offset:
                _clear_kv_cache = True
                # Assistant
                if conv.messages[-1][0] == conv.roles[1]:
                    conv.messages.pop()
                # User
                if conv.messages[-1][0] == conv.roles[0]:
                    reload_conv(conv)
                    # Set inp to previous message
                    inp = conv.messages.pop()[1]
                else:
                    # Shouldn't happen in normal circumstances
                    print("No user message to regenerate from.")
                    continue
            else:
                print("No messages to regenerate.")
                continue
        elif inp.startswith("!!save"):
            args = inp.split(" ", 1)

            if len(args) != 2:
                print("usage: !!save <filename>")
                continue
            else:
                filename = args[1]

            # Add .json if extension not present
            if not "." in filename:
                filename += ".json"

            print("saving...", filename)
            with open(filename, "w") as outfile:
                json.dump(conv.dict(), outfile)
            continue
        elif inp.startswith("!!load"):
            args = inp.split(" ", 1)

            if len(args) != 2:
                print("usage: !!load <filename>")
                continue
            else:
                filename = args[1]

            # Check if file exists and add .json if needed
            if not os.path.exists(filename):
                if (not filename.endswith(".json")) and os.path.exists(
                    filename + ".json"
                ):
                    filename += ".json"
                else:
                    print("file not found:", filename)
                    continue

            print("loading...", filename)
            with open(filename, "r") as infile:
                new_conv = json.load(infile)

            conv = get_conv_template(new_conv["template_name"])
            conv.set_system_message(new_conv["system_message"])
            conv.messages = new_conv["messages"]
            reload_conv(conv)
            _clear_kv_cache = True
            continue

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if is_codet5p:  # codet5p is a code completion model.
            prompt = inp

        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
            "top_k": top_k,
            "top_p": top_p
        }

        try:
            chatio.prompt_for_output(conv.roles[1])
            output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
                judge_sent_end=judge_sent_end,
                clear_kv_cache=_clear_kv_cache
            )
            t = time.time()
            outputs = chatio.stream_output(output_stream)
            duration = time.time() - t
            conv.update_last_message(outputs.strip())
            _clear_kv_cache = clear_kv_cache

            if debug:
                num_tokens = len(tokenizer.encode(outputs))
                msg = {
                    "conv_template": conv.name,
                    "prompt": prompt,
                    "outputs": outputs,
                    "speed (token/s)": round(num_tokens / duration, 2),
                }
                print(f"\n{msg}\n")

        except KeyboardInterrupt:
            print("stopped generation.")
            # If generation didn't finish
            if conv.messages[-1][1] is None:
                conv.messages.pop()
                # Remove last user message, so there isn't a double up
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()

                reload_conv(conv)

                
def main(args):
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ["XPU_VISIBLE_DEVICES"] = args.gpus
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None

    if args.inf_llm_config_path is not None:
        from omegaconf import OmegaConf
        inf_llm_config = OmegaConf.load(args.inf_llm_config_path)
        if inf_llm_config.conv_type == "llama-3-inst":
            args.conv_template = "llama-3-inst"

        inf_llm_config = inf_llm_config["model"]
    else:
        inf_llm_config = None

    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None
    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        chat_loop(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            str_to_torch_dtype(args.dtype),
            args.load_8bit,
            args.cpu_offloading,
            args.conv_template,
            args.conv_system_msg,
            args.temperature,
            args.repetition_penalty,
            args.max_new_tokens,
            chatio,
            gptq_config=GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            ),
            awq_config=AWQConfig(
                ckpt=args.awq_ckpt or args.model_path,
                wbits=args.awq_wbits,
                groupsize=args.awq_groupsize,
            ),
            exllama_config=exllama_config,
            xft_config=xft_config,
            revision=args.revision,
            judge_sent_end=args.judge_sent_end,
            debug=args.debug,
            history=not args.no_history,
            inf_llm_config=inf_llm_config,
            clear_kv_cache=args.clear_kv_cache
        )
    except KeyboardInterrupt:
        print("exit...")

from fastchat.conversation import Conversation, register_conv_template, SeparatorStyle
import dataclasses

@dataclasses.dataclass
class Llama3Conv(Conversation):
    # role format
    role_format: str = "{role}"

    def get_prompt(self) -> str:
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = system_prompt
        for role, message in self.messages:
            if message:
                ret += self.role_format.format(role=role) + message + self.sep
            else:
                ret += self.role_format.format(role=role)
        return ret
    
    def copy(self):
        return Llama3Conv(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            role_format=self.role_format
        )

register_conv_template(
    Llama3Conv(
        "llama-3-inst",
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|eot_id|>",
        role_format="<|start_header_id|>{role}<|end_header_id|>\n\n",
        stop_token_ids=[128009, 128001]
    )
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    parser.add_argument(
        "--inf-llm-config-path",
        type=str, help="Inf LLM patch config",
        default=None
    )
    parser.add_argument(
        "--clear-kv-cache",
        action="store_true"
    )
    args = parser.parse_args()
    main(args)
