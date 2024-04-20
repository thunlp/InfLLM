# https://github.com/THUDM/LongBench/blob/main/pred.py
import os
from datasets import load_from_disk
import torch
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from inf_llm.utils import patch_hf, GreedySearch, patch_model_center
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--model_center", action="store_true", default=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.output_dir_path = args.output_dir_path
    conf.model.model_center = args.model_center
    conf.rank = args.rank
    conf.world_size = args.world_size
    conf.verbose = args.verbose
    if not hasattr(conf.model, "tokenizer_path"):
        conf.model.tokenizer_path = conf.model.path
    if not hasattr(conf, "truncation"):
        conf.truncation = None

    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    conf.datasets = []
    for d in datasets_list:
        conf.datasets.append(d.strip())
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
        model = AutoModelForCausalLM.from_pretrained(config.path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
        model = patch_hf(model, config.type, **config)
    return model, tokenizer

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    model_name = model_name.strip().lower()
    if model_name == "vicuna":
        from fastchat.conversation import get_conv_template
        conv = get_conv_template("vicuna_v1.1")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif model_name in ["mistral-inst", "qwen", "minicpm", "llama-3-inst"]:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        raise NotImplementedError

    return prompt

def load_infinite_bench(path, data_name) -> str:
    import re
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
    """
    print(f"read {data_name}.jsonl")
    fin = open(os.path.join(path, data_name + ".jsonl"), "r")
    lines = fin.readlines()
    fin.close()
    data = [json.loads(line) for line in lines]
    def get_answer(inp: dict):
        if data_name in ["code_debug", "longbook_choice_eng"]:
            OPTIONS = "ABCD"
            if isinstance(inp["answer"], str):
                ret = [inp["answer"], OPTIONS[inp['options'].index(inp["answer"])]]
            elif isinstance(inp["answer"], list):
                if len(inp["answer"]) == 1:
                    ret = [inp["answer"][0], OPTIONS[inp['options'].index(inp["answer"][0])]]
                elif len(inp["answer"]) == 2 and inp["answer"][1] in ['A', 'B', 'C', 'D']:
                    ret = inp['answer']
                else:
                    raise ValueError
            else:
                raise ValueError
            return ret
        return inp["answer"]

    ret = []
    for eg in data:
        # ================= Code tasks
        if data_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            instance = {"func": func, "func_call": func_call, "context": eg["context"]}
        elif data_name in ["code_debug", "code_debug_qa"]:
            # Load source code
            instance = {"context": eg["context"]}
            if data_name == "code_debug":
                instance.update({
                    "OPTION_A": eg["options"][0], 
                    "OPTION_B": eg["options"][1], 
                    "OPTION_C": eg["options"][2], 
                    "OPTION_D": eg["options"][3]})
        # ================= Code tasks
        elif data_name == "longdialogue_qa_eng":
            instance = {"context": eg["context"]}
        # ==================== Long book tasks
        elif data_name in [
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_sum_eng",
            "longbook_qa_chn",
        ]:
            instance = {"context": eg["context"]}
            if data_name == "longbook_choice_eng":
                instance.update({
                    "question": eg["input"],
                    "OPTION_A": eg["options"][0],
                    "OPTION_B": eg["options"][1],
                    "OPTION_C": eg["options"][2],
                    "OPTION_D": eg["options"][3],
                })
            elif data_name in ["longbook_qa_eng", "longbook_qa_chn"]:
                instance.update({
                    "question": eg["input"],
                })
        elif data_name == "math_calc":
            instance = {"context": eg["context"]}
        elif data_name == "math_find":
            prompt = eg['input']
            context = eg['context']
            # Find "the * number" from the prompt
            find_result = re.findall(r"The .+ of", prompt)
            assert find_result, f"Cannot find the target number in {prompt}"
            target_number = find_result[0].lower()[:-3]
            # Replace the number with the answer
            prefix = f"What is {target_number} in the following list?"
            instance = {"prefix": prefix, "context": context, "input": prompt}
        elif data_name == "kv_retrieval":
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
                "key": eg["input"][6:44]
            }
            assert eg['input'][6] == '"'
            assert eg['input'][43] == '"'
        else:
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
            }
        ans = get_answer(eg)
        instance["answers"] = ans if isinstance(ans, list) else [ans]
        instance["length"] = len(instance["context"].split())
        instance["all_classes"] = None
        
        ret.append(instance)
        # if len(ret) > 4:
        #     break
    return ret

def post_process(pred, model_name, dataset):
    if model_name == "qwen":
        pred = pred.split("<|im_end|>")[0]

    if dataset == "samsum":
        pred = pred.split("\n")[0].strip()

    return pred

def get_pred(
    model, tokenizer, data, max_length,
    max_gen, prompt_format, dataset, model_name, 
    gen_chunk_size = None, truncation: str = None, 
    rank: int = None, world_size: int = None,
    verbose: bool = False
):
    preds = []
    data = list(data)

    if world_size is not None:
        data = data[rank::world_size]

    searcher = GreedySearch(model, tokenizer)
    cur = 0
    total = len(data)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        extra_end_token_ids = []
        if model_name == "llama-3-inst":
            extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])

        if model_name == "qwen":
            extra_end_token_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])

        if dataset == "samsum":
            extra_end_token_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: 
            # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

            if model_name.strip().lower() in ['mistral-inst']:
                add_special_tokens = False
            else:
                add_special_tokens = True
        
        else:
            add_special_tokens = True

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]

        if truncation is None:
            if len(tokenized_prompt) > max_length - max_gen:
                if verbose:
                    print(f"Length {len(tokenized_prompt)}. Skipped.")
                continue

        else:
            if truncation == "suffix":
                length = len(tokenized_prompt)
                if length > max_length - max_gen:
                    if verbose:
                        print("over length")
                    init_token_num = 128
                    prompt = tokenizer.decode(tokenized_prompt[:init_token_num].tolist() + tokenized_prompt[- (max_length - max_gen - init_token_num):].tolist())
                    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
            else:
                raise NotImplementedError
    

        
        output = searcher.generate(
            input_ids = tokenized_prompt,
            max_length=max_gen,
            chunk_size=gen_chunk_size,
            extra_end_token_ids=extra_end_token_ids
        )

        pred = post_process(output[0], model_name, dataset)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "token_length": len(tokenized_prompt) + max_gen})
        searcher.clear()
        cur += 1
        if verbose:
            print(f"----------{cur}/{total}----------")
            print("Length: ", len(tokenized_prompt))
            print("Question:", prompt[-100:])
            print("Pred:", pred)
            print("Answer:", json_obj["answers"])
            print("")


    return preds


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model
    model, tokenizer = get_model_and_tokenizer(args.model)
    output_dir_path = args.output_dir_path

    datasets = args.datasets


    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("benchmark/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/config/dataset2maxlen.json", "r"))

    
    multiprocessing = args.world_size is not None and args.world_size > 1
    if multiprocessing:
        assert args.rank in list(range(args.world_size))

    # predict on each dataset
    for dataset in datasets:
        dname = dataset
        if dataset in set([
            "kv_retrieval", "passkey", "number_string", "code_run", "code_debug", "longdialogue_qa_eng", "longbook_qa_eng", "longbook_sum_eng", "longbook_choice_eng", "longbook_qa_chn", "math_find", "math_calc"
        ]):
            path = "benchmark/data/infinite-bench"
            data = load_infinite_bench(path, dname)

        else:
            data = load_from_disk(
                f"benchmark/data/longbench/{dataset}"
            )

        out_path = os.path.join(
            output_dir_path,
            f"{dname}.jsonl"
        )

        print(f"Pred {dname}")
        prompt_format = dataset2prompt[dataset]

        max_gen = dataset2maxlen[dataset]
        preds = get_pred(
            model, tokenizer, data, 
            args.max_len, max_gen, 
            prompt_format, dataset, 
            args.conv_type, 
            args.chunk_size, args.truncation,
            args.rank, args.world_size,
            args.verbose
        )
        if multiprocessing:
            out_path = out_path + f"_{args.rank}"
        with open(out_path, "w+", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')