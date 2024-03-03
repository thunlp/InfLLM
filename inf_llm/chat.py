from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import argparse
from omegaconf import OmegaConf
from fastchat.conversation import get_conv_template
from inf_llm.utils.patch import patch_hf
from inf_llm.utils.greedy_search import GreedySearch
from inf_llm.utils.patch_mc import patch_model_center

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--model_center", action="store_true", default=False)
    parser.add_argument("--max_gen", default=None, type=int)
    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.model.model_center = args.model_center
    if args.max_gen is not None:
        conf.max_gen = args.max_gen
    if not hasattr(conf.model, "tokenizer_path"):
        conf.model.tokenizer_path = conf.model.path

    return conf
    

def get_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    if config.model_center:
        import bmtrain as bmt
        bmt.init_distributed(seed=233)
        from model_center.model import Llama, LlamaConfig
        model_config = LlamaConfig.from_pretrained(config.path)
        model_config.dtype = torch.bfloat16
        model = Llama(model_config)
        bmt.load(model, os.path.join(config.path, "pytorch_model.pt"), strict=False)
        model = patch_model_center(model, config.type, **config)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.path).bfloat16().cuda()
        model = patch_hf(model, config.type, **config)
    return model, tokenizer

class Conversation:
    def __init__(self, model, tokenizer, max_gen, chunk_size):
        self.messages = []
        self.searcher = GreedySearch(model, tokenizer)
        self.tokens = []
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_gen = max_gen

    def get_tokenized_prompt(self, add_generation_prompt):
        raise NotImplementedError

    def append(self, text) -> str:
        self.messages.append(
            {
                "role": "user",
                "content": text
            }
        )
        tokenized_prompt = self.get_tokenized_prompt(True)
        new_tokens = torch.tensor(tokenized_prompt[len(self.tokens):])
        gen_text = self.searcher.generate(input_ids = new_tokens, max_length=self.max_gen, chunk_size=self.chunk_size, output=True)[0]
        self.messages.append(
            {
                "role": "assistant",
                "content": gen_text
            }
        )
        self.tokens = self.get_tokenized_prompt(False)
        return gen_text


    def clear(self):
        self.messages = []
        self.searcher.clear()
        self.tokens = []

class MistralConv(Conversation):
    def get_tokenized_prompt(self, add_generation_prompt):
        tokenized_prompt = self.tokenizer.apply_chat_template(self.messages, tokenize=True, add_generation_prompt=add_generation_prompt)
        return tokenized_prompt
        

class VicunaConv(Conversation):
    def get_tokenized_prompt(self, add_generation_prompt):
        conv = get_conv_template("vicuna_v1.1")
        for i, msg in enumerate(self.messages):
            if i % 2 == 0:
                conv.append_message(conv.roles[0], msg["content"])
            else:
                conv.append_message(conv.roles[1], msg["content"])

        if add_generation_prompt:
            conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        return self.tokenizer(prompt).input_ids[0]

CONV = {
    "mistral-inst": MistralConv,
    "vicuna": VicunaConv
}

def chat(config):
    model, tokenizer = get_model_and_tokenizer(config.model)
    conv = CONV[config.conv_type](model, tokenizer, config.get("max_gen", 4096), config.get("chunk_size", 4096))
    print("USER:")
    while True:
        t = input()
        print("ASSISTANT:")
        conv.append(t)
        print("USER:")

if __name__ == "__main__":
    config = parse_args()
    chat(config)