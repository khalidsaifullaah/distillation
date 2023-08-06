"""This script generates the evaluation responses that can e used by eval_scoring.py"""
import argparse
from functools import partial
import json
import pickle
import random
import time
from typing import Dict
import gc

from datasets import Dataset
import datasets
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.models.llama.configuration_llama import LlamaConfig
import torch
from tqdm import tqdm
import copy
import train_AL
import subprocess

from io_utils import load_jsonlines
from utils import load_fsdp_ckpt_with_accelerate
from conversation import get_conv_template

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def apply_conv_template(example):
    # preprocess instructions into prompted inputs
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    source  = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)

    example.update({
        "prompt": source
    })

    return example

def get_clm_loss(labels, lm_logits):
    # move labels to correct device to enable model parallelism
    labels = labels.to(lm_logits.device)
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # get per-token loss
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss_per_example = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    return loss_per_example

def compute_perplexity_batched(example, model, tokenizer, kwargs):
    prompt = example['prompt']
    encoding = tokenizer(prompt, 
                          return_tensors="pt",
                          padding="longest",
                          max_length=tokenizer.model_max_length,
                          truncation=True,
                      )
    encoding.pop('token_type_ids', None)
    encoding = {k: v.cuda() for k,v in encoding.items()}
    # labels = copy.deepcopy(encoding['input_ids'])
    # labels_lens = torch.sum(labels != tokenizer.pad_token_id, dim=-1).tolist()
    # labels[labels == tokenizer.pad_token_id] = IGNORE_INDEX
    with torch.no_grad():
        model_output = model(**encoding)
        if torch.isnan(model_output.logits).any():
            print("NAN")
        loss = get_clm_loss(encoding['input_ids'], model_output.logits.float())
        # model_output = model.generate(**encoding, **kwargs)
        # logits = torch.nan_to_num(torch.stack(model_output.scores).transpose(0,1))
        # labels = model_output.sequences[:, encoding['input_ids'].shape[-1]:]
        # loss = get_clm_loss(labels, logits.float())
        ppl = torch.exp(loss).tolist()
        # input_len = encoding['input_ids'].shape[-1]
        # output_sequences = model_output.sequences[:, input_len:].cpu()
        # decoded_output = tokenizer.batch_decode(output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    del example['prompt']
    example.update({"ppl": ppl})
    # example.update({"decoded_output": decoded_output})

    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ksaifullah/llama2_7B_sharded", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--model_config_path", default="/home/ksaifullah/llama2_7B_hf", type=str)
    parser.add_argument("--template_type", default="alpaca", type=str)
    parser.add_argument("--file_path", default="datasets/dolphin.jsonl", type=str)
    parser.add_argument("--save_file_name", default="outputs/al_train_dataset.jsonl", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--cluster_data_fraction", default=0.001, type=float)
    parser.add_argument("--al_data_fraction", default=0.001, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--debug", action='store_true', help="This reduce the number of generation examples to 4, so that we can debug faster.")
    args = parser.parse_args()

    # model_config = transformers.AutoConfig.from_pretrained(args.model_config_path)
    # if isinstance(model_config, LlamaConfig):
    #     model_config.vocab_size += 1 # hardcode the vocab size for llama...

    # model = load_fsdp_ckpt_with_accelerate(args.model, model_config, hf_dummy_path=args.model_config_path, wrapped_class="LlamaDecoderLayer" if 'llama' in args.model else "OPTDecoderLayer")
    pool_data = datasets.load_dataset('json', data_files=args.file_path, split='train')
    pool_data = pool_data.add_column('id', list(range(len(pool_data))))
    sampled_data = pool_data.shuffle(seed=args.seed).select(range(1000))

    used_data_count = int(len(pool_data)*args.cluster_data_fraction)
    al_data_count = int(len(pool_data)*args.al_data_fraction)

    initial_run = True
    # track loop time
    start_time = time.time()
    while len(sampled_data) < al_data_count:
        if initial_run:
            # saving sampled data
            sampled_data.to_json(args.save_file_name)
            print("#"*100)
            print("Running initial training")
            print(f"sampled_data size: {len(sampled_data)}, total data: {al_data_count}")
            print("#"*100)
            cmd = ["python", "train_AL.py"]
            cmd.extend(["--init_checkpoint_path", "/home/ksaifullah/llama2_7B_sharded", "--model_config_path", "/home/ksaifullah/llama2_7B_hf", "--checkpoint_path", f"/home/ksaifullah/al_dolphin_llama2_7B_dfrac_{args.al_data_fraction}_ppl", "--wrapped_class_name", "LlamaDecoderLayer", "--data_path", args.save_file_name, "--hack", "--batch_size", "1", "--accumulation_steps", "8", "--dont_save_opt"]) # , "--wandb", "--wb_name", f"al_dolphin_llama2_7B_dfrac_{args.al_data_fraction}_ppl"
            result = subprocess.run(cmd)
            initial_run = False

        else:
            print("#"*100)
            print("Sampling new data")
            print(f"sampled_data size: {len(sampled_data)}, total data: {al_data_count}")
            print("#"*100)
            model = transformers.AutoModelForCausalLM.from_pretrained(f"/home/ksaifullah/al_dolphin_llama2_7B_dfrac_{args.al_data_fraction}_ppl", torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                    f"/home/ksaifullah/al_dolphin_llama2_7B_dfrac_{args.al_data_fraction}_ppl",
                    model_max_length=2048,
                    padding_side="left",
                    use_fast=True,
                )
            special_tokens_dict = dict()
            if tokenizer.pad_token is None:
                special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
            if tokenizer.eos_token is None:
                special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
            if tokenizer.bos_token is None:
                special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
            if tokenizer.unk_token is None:
                special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                tokenizer=tokenizer,
                model=model,
            )

            ## set the models to eval mode
            model = model.eval()
            #remove the sampled data from the pool data with .filter()
            # pool_data.set_format('pandas')
            # df = pool_data[:]
            # df = df[~df.index.isin(sampled_data['id'])]
            # pool_data = Dataset.from_pandas(df, preserve_index=False)

            with open('clusters.pkl', 'rb') as f:
                clusters = pickle.load(f)
            random.seed(args.seed)
            sampled_clusters = random.choices(list(clusters.keys()), k=used_data_count)
            cluster_sampling_data = {
                'id': [],
                'instruction': [],
                'input': [],
                'output': []
            }
            
            # sampled_data = datasets.load_dataset('json', data_files=args.save_file_name, split='train')
            sampled_data_ids = set(sampled_data['id'])
            for c in tqdm(sampled_clusters):
                idx = random.sample(range(len(clusters[c])), 1)[0]
                sample_id = int(clusters[c][idx][0])  # getting the int from numpy.int64
                while sample_id in sampled_data_ids:  # we want to get a fresh sample from pool that isn't already present in the training set
                    sample_cluster = random.sample(list(clusters.keys()), 1)[0]
                    idx = random.sample(range(len(clusters[sample_cluster])), 1)[0]
                    sample_id = int(clusters[sample_cluster][idx][0])
                cluster_sampling_data['id'].append(pool_data[sample_id]['id'])
                cluster_sampling_data['instruction'].append(pool_data[sample_id]['instruction'])
                cluster_sampling_data['input'].append(pool_data[sample_id]['input'])
                cluster_sampling_data['output'].append(pool_data[sample_id]['output'])

            cluster_sampling_data = Dataset.from_dict(cluster_sampling_data)

            ## preprocess
            eval_preproc = partial(apply_conv_template)
            cluster_sampling_data = cluster_sampling_data.map(eval_preproc)
            # cluster_sampling_data = cluster_sampling_data.map(lambda examples: {"prompt": [examples['instruction'][i]+' '+examples['input'][i] for i in range(len(examples['input']))]}, batched=True)

            generate_kwargs = dict(max_new_tokens=256, do_sample=False,
                                num_return_sequences=1, output_scores=True, return_dict_in_generate=True, 
                                stopping_criteria=StoppingCriteriaList([StopOnTokens()]))
            get_ppl = partial(compute_perplexity_batched, 
                            model=model,  
                            tokenizer=tokenizer,
                            kwargs=generate_kwargs)
            dataset_w_ppl = cluster_sampling_data.map(get_ppl,
                                                    batched=True,
                                                    batch_size=args.batch_size)
            dataset_w_ppl = dataset_w_ppl.sort('ppl', reverse=True)
            model = model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            # we don't want to add more data than the total data count
            if len(sampled_data)+1000 <= al_data_count:
                acquisition_samples = dataset_w_ppl.select(range(1000))
            else:
                acquisition_samples = dataset_w_ppl.select(range(al_data_count-len(sampled_data)))
            acquisition_samples = acquisition_samples.remove_columns('ppl')
            sampled_data = datasets.concatenate_datasets([sampled_data, acquisition_samples])
            sampled_data.to_json(args.save_file_name)
            print("#"*100)
            print("Training on new data")
            print(f"sampled_data size: {len(sampled_data)}, total data: {al_data_count}")
            print("#"*100)
            cmd = ["python", "train_AL.py"]
            cmd.extend(["--init_checkpoint_path", "/home/ksaifullah/llama2_7B_sharded", "--model_config_path", "/home/ksaifullah/llama2_7B_hf", "--checkpoint_path", f"/home/ksaifullah/al_dolphin_llama2_7B_dfrac_{args.al_data_fraction}_ppl", "--wrapped_class_name", "LlamaDecoderLayer", "--data_path", args.save_file_name, "--hack", "--batch_size", "1", "--accumulation_steps", "8", "--dont_save_opt"]) # , "--wandb", "--wb_name", f"al_dolphin_llama2_7B_dfrac_{args.al_data_fraction}_ppl"
            result = subprocess.run(cmd)

    end_time = time.time()
    print(f"Total time taken: {(end_time-start_time)/60} minutes")