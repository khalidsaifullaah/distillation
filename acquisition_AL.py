"""This script generates the evaluation responses that can e used by eval_scoring.py"""
import argparse
from functools import partial
import json
import pickle
import random

from datasets import Dataset
import datasets
import transformers
from transformers.models.llama.configuration_llama import LlamaConfig
import torch

from io_utils import load_jsonlines
from utils import load_fsdp_ckpt_with_accelerate
from conversation import get_conv_template


def add_padding_token(tokenizer):
    print("attempt to add padding token if no padding token exists")
    print("Special tokens before adding padding token: ", tokenizer.special_tokens_map)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print("Special tokens after adding padding token: ", tokenizer.special_tokens_map)
    return tokenizer

def apply_conv_template(example, template_type):
    # preprocess instructions into prompted inputs
    conv = get_conv_template(template_type)
    conv.append_message(conv.roles[0], example['instruction'])
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    example.update({
        "prompt": prompt
    })

    return example

def compute_perplexity_batched(example, model, tokenizer):
    prompt = example['prompt']
    print(prompt)
    encoding = tokenizer(prompt, 
                          return_tensors="pt",
                          padding="longest",
                          max_length=tokenizer.model_max_length,
                          truncation=True,
                      )
    encoding = encoding.to(model.device)
    with torch.no_grad():
        model_output = model(**encoding)
        from IPython import embed; embed()
        model_output = model_output[:, input_len:].cpu()
        ppl = 0

    del example['prompt']
    example.update({"ppl": ppl}) 

    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ksaifullah/llama2_7B_sharded", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--model_config_path", default="/home/ksaifullah/llama2_7B_hf", type=str)
    parser.add_argument("--template_type", default="alpaca", type=str)
    parser.add_argument("--file_path", default="datasets/dolphin.jsonl", type=str)
    parser.add_argument("--save_file_name", default="outputs/answers/self-instruct_llama7B.jsonl", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_fraction", default=0.05, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--debug", action='store_true', help="This reduce the number of generation examples to 4, so that we can debug faster.")
    args = parser.parse_args()

    model_config = transformers.AutoConfig.from_pretrained(args.model_config_path)
    if isinstance(model_config, LlamaConfig):
        model_config.vocab_size += 1 # hardcode the vocab size for llama... 

    model = load_fsdp_ckpt_with_accelerate(args.model, model_config, hf_dummy_path=args.model_config_path, wrapped_class="LlamaDecoderLayer" if 'llama' in args.model else "OPTDecoderLayer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_config_path,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,
        )
    add_padding_token(tokenizer)
    
    ## set the models to eval mode
    model = model.eval()
    
    pool_data = datasets.load_dataset('json', data_files=args.file_path, split='train')
    # creating a new column for the index of the example
    pool_data = pool_data.add_column('id', list(range(len(pool_data))))
    # randomly sampling 1000 examples from the pool data
    sampled_data = pool_data.shuffle(seed=args.seed).select(range(1000))
    #remove the sampled data from the pool data with .filter()
    pool_data.set_format('pandas')
    df = pool_data[:]
    df = df[~df.index.isin(sampled_data['id'])]
    pool_data = Dataset.from_pandas(df, preserve_index=False)

    used_data_count = int(len(pool_data)*args.data_fraction)
    with open('clusters.pkl', 'rb') as f:
        clusters = pickle.load(f)
    random.seed(args.seed)
    sampled_clusters = random.choices(list(clusters.keys()), k=used_data_count)
    filtered_data = {
        'instruction': [],
        'input': [],
        'output': []
    }
    for c in sampled_clusters:
        idx = random.sample(range(len(clusters[c])), 1)[0]
        sample_id = int(clusters[c][idx][0])  # getting the int from numpy.int64
        filtered_data['instruction'].append(pool_data[sample_id]['instruction'])
        filtered_data['input'].append(pool_data[sample_id]['input'])
        filtered_data['output'].append(pool_data[sample_id]['output'])

    pool_data = Dataset.from_dict(filtered_data)

    from IPython import embed; embed()

    # if "dolly" in args.file_path or "vicuna" in args.file_path or "user_oriented_instructions" in args.file_path:
    #     tasks = load_jsonlines(args.file_path)
    #     raw_data = Dataset.from_list(tasks)
    #     if "dolly" in args.file_path:
    #         question_sources = "dolly"
    #     elif "vicuna" in args.file_path:
    #         question_sources = "vicuna"
    #         raw_data = raw_data.rename_column("text", "instruction")
    #     elif "user_oriented_instructions" in args.file_path:
    #         question_sources = "alpaca"
    # elif "alpaca_eval" in args.file_path:
    #     raw_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    #     # set generator field to model name
    #     raw_data = raw_data.map(lambda x: {"generator": args.model_name if args.model_name else args.model})

    ## preprocess
    eval_preproc = partial(apply_conv_template, template_type=args.template_type)
    pool_data = pool_data.map(eval_preproc)

    ## run generation
    # generate_kwargs = dict(max_length=1024, do_sample=False, top_p=0.1, 
    #                        num_return_sequences=1, temperature=0.1, 
    #                        repetition_penalty=1.2)
    get_ppl = partial(compute_perplexity_batched, 
                       model=model,  
                       tokenizer=tokenizer)

    dataset_w_responses = pool_data.map(get_ppl,
                                            batched=True,
                                            batch_size=args.batch_size)
    if "alpaca_eval" in args.file_path:
        # stringify the metadata, so that it becomes hashable
        dataset_w_responses = dataset_w_responses.map(lambda x: {"metadata": json.dumps(x["metadata"])})
        dataset_w_responses.to_json(args.save_file_name, orient="records", lines=False, indent=True)
        
    else:
        dataset_w_responses.to_json(args.save_file_name)