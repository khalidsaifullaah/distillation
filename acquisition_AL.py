"""This script generates the evaluation responses that can e used by eval_scoring.py"""
import os
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
import torch.multiprocessing as mp
from tqdm import tqdm
import copy
import train_AL
import subprocess
import matplotlib.pyplot as plt

from io_utils import load_jsonlines
from utils import load_fsdp_ckpt_with_accelerate, add_padding_token
from conversation import get_conv_template

from dataset import DataCollatorForSupervisedDataset

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
        "### Instruction:\n{input}\n\n### Response:"
    ),
    "confidence_prompt": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nHere's a user query: '{input}'\nNow, looking at the user query respond whether you know the answer or not (Please only respond with 'yes' or 'no')\n\n### Response:"
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
    # if args.sampling_strategy == "generation_ppl":
    #     model.resize_token_embeddings(32008) # making a multiple of 8
    # else:
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
    prompt_input, prompt_no_input, confidence_prompt = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"], PROMPT_DICT["confidence_prompt"]
    if args.sampling_strategy != "self_ask":
        source  = prompt_input.format_map(example) if example.get("instruction", "") != "" else prompt_no_input.format_map(example)
    else:
        source = confidence_prompt.format_map(example)
    example.update({
        "prompt": source
    })

    return example

def get_clm_loss(labels, lm_logits, tokenizer=None):
    # move labels to correct device to enable model parallelism
    labels = labels.to(lm_logits.device)
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    if IGNORE_INDEX not in labels:
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
    else:
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # get per-token loss
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss_per_example = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    return loss_per_example

def compute_perplexity_batched(example, model, tokenizer, device=0, kwargs=None):
    prompt, targets = example['prompt'], example['output']
    if args.sampling_strategy == "data_pruning_w_answers":
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        prompts_w_answers_encoding = data_collator([{'sources': source, 'targets': target} for source,target in zip(prompt,targets)])
        prompts_w_answers_encoding = {k: v.to(f"cuda:{device}") for k,v in prompts_w_answers_encoding.items()}
        target_encoding = tokenizer(targets, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True)
        target_encoding.pop('token_type_ids', None)
        target_encoding = {k: v.to(f"cuda:{device}") for k,v in target_encoding.items()}
    else:
        encoding = tokenizer(prompt, 
                            return_tensors="pt",
                            padding="longest",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                        )
        encoding.pop('token_type_ids', None)
        encoding = {k: v.to(f"cuda:{device}") for k,v in encoding.items()}
    with torch.no_grad():
        if args.sampling_strategy == "forward_ppl":
            model_output = model(**encoding)
            if torch.isnan(model_output.logits).any():
                print("NAN")
            loss = get_clm_loss(encoding['input_ids'], model_output.logits.float(), tokenizer=tokenizer)
            log_ppl = loss.tolist()
        elif args.sampling_strategy == "data_pruning_w_answers":
            # first forward pass with prompt responses
            model_output_1 = model(**target_encoding)
            # second forward pass with prompt + responses
            model_output_2 = model(input_ids = prompts_w_answers_encoding['input_ids'], attention_mask = prompts_w_answers_encoding['attention_mask'])
            loss_on_response = get_clm_loss(target_encoding['input_ids'], model_output_1.logits.float(), tokenizer=tokenizer)
            loss_on_response_given_instruction = get_clm_loss(prompts_w_answers_encoding['labels'], model_output_2.logits.float(), tokenizer=tokenizer)
            # get absolute difference between the two losses
            loss = torch.abs(loss_on_response - loss_on_response_given_instruction)
            log_ppl = loss.tolist()
        elif args.sampling_strategy == "generation_ppl":
            model_output = model.generate(**encoding, **kwargs)
            # logits = torch.nan_to_num(torch.stack(model_output.scores).transpose(0,1))
            logits = torch.stack(model_output.scores).transpose(0,1)
            if torch.isnan(logits).any():
                print("NAN")
            labels = model_output.sequences[:, encoding['input_ids'].shape[-1]:]
            loss = get_clm_loss(labels, logits.float()) # using log perplexity
            log_ppl = loss.tolist()
        elif args.sampling_strategy == "self_ask":
            kwargs['max_new_tokens'] = 3
            model_output = model.generate(**encoding, **kwargs)
            logits = torch.nan_to_num(torch.stack(model_output.scores).transpose(0,1))
            token_probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence = token_probs[:, :, torch.tensor([1939,694])].view(len(prompt), -1).mean(dim=-1).tolist() # 1939: No, 694: no
            del example['prompt']
            # input_len = encoding['input_ids'].shape[-1]
            # output_sequences = model_output.sequences[:, input_len:].cpu()
            # decoded_output = tokenizer.batch_decode(output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return confidence
            # return decoded_output, 
        else:
            print("Invalid sampling strategy")
            raise NotImplementedError

    del example['prompt']
    # example.update({"ppl": log_ppl})
    # example.update({"decoded_output": decoded_output})
    # return example
    return log_ppl

def inference_worker(rank, sharded_model_path, data_partition, result_list):
    from optimum.bettertransformer import BetterTransformer
    gpu = rank
    model_config = transformers.AutoConfig.from_pretrained(args.model_config_path)
    model = load_fsdp_ckpt_with_accelerate(sharded_model_path, model_config, hf_dummy_path=args.model_config_path, wrapped_class="LlamaDecoderLayer", rank=gpu)
    model.eval()
    # model = BetterTransformer.transform(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_config_path,
            model_max_length=args.model_max_length,
            # padding_side="left",
            padding_side="right",
            use_fast=True,
            # pad_to_multiple_of=8,
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
    generate_kwargs = dict(max_new_tokens=256, do_sample=False, temperature=0.7,
        num_return_sequences=1, output_scores=True, return_dict_in_generate=True,
        # stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    get_ppl = partial(compute_perplexity_batched, 
                    model=model,  
                    tokenizer=tokenizer,
                    device=gpu,
                    kwargs=generate_kwargs)

    ppl_list = []
    for batch in tqdm(data_partition.iter(batch_size=args.batch_size)):
        batch_ppl = get_ppl(batch)
        ppl_list.extend(batch_ppl)
    dataset_w_ppl = data_partition.add_column('ppl', ppl_list)
    result_list.append(dataset_w_ppl)
    # model = model.cpu()

    return


def main(args):
    # torch.set_num_threads(1)
    pool_data = datasets.load_dataset('json', data_files=args.file_path, split='train')
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         args.model_config_path,
    #         model_max_length=args.model_max_length,
    #         padding_side="right",
    #         use_fast=True,
    # )
    # special_tokens_dict = dict()
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # if tokenizer.bos_token is None:
    #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # if tokenizer.unk_token is None:
    #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    # tokenizer.add_special_tokens(special_tokens_dict)
    # eval_preproc = partial(apply_conv_template)
    # pool_data = pool_data.map(eval_preproc)
    # pool_data = pool_data.filter(lambda x: len(tokenizer(x['prompt']).input_ids) < args.model_max_length)
    pool_data = pool_data.add_column('id', list(range(len(pool_data))))
    pool_data = pool_data.shuffle(seed=args.seed)
    pool_data_count = int(len(pool_data)*args.cluster_data_fraction)
    al_data_count = int(len(pool_data)*args.al_data_fraction)
    sampled_data = pool_data.select(range(args.num_acquisition_samples))
    model_path = args.model_path if args.model_path else f"/home/ksaifullah/al_dolphin_llama2_7B_dfrac_{args.al_data_fraction}_poolfrac_{args.cluster_data_fraction}_forward_ppl"
    
    initial_run = True
    resume = args.resume
    if resume:
        initial_run = False
    steps = 0
    start_time = time.time()
    while len(sampled_data) < al_data_count:
        if initial_run:
            # saving sampled data
            sampled_data.to_json(f"{args.save_file_name}")
            print("#"*100)
            print("Running seed training")
            print(f"steps: {steps}, sampled_data size: {len(sampled_data)}, total data: {al_data_count}")
            print("#"*100)
            cmd = ["python", "train_AL.py"]
            cmd.extend(["--init_checkpoint_path", f"{args.init_checkpoint_path}", "--model_config_path", f"{args.model_config_path}", "--checkpoint_path", f"{model_path}_sharded", "--wrapped_class_name", "LlamaDecoderLayer", "--data_path", f"{args.save_file_name}", "--hack", "--batch_size", "1", "--accumulation_steps", "8", "--dont_save_opt", "--num_epochs", "2", "--lr", f"{args.lr}", , "--seed", f"{args.seed}", "--filtering_method", "random"]) # , "--wandb", "--wb_name", f"s_{steps}_{model_path.split('/')[-1]}", "--wb_project", "al_data_distillation"
            result = subprocess.run(cmd)
            initial_run = False
        else:
            if resume:
                sharded_model_path = f"{args.resume_checkpoint_path}/{os.listdir(args.resume_checkpoint_path)[0]}/model"
                prev_data_fraction = args.resume_checkpoint_path.split('dfrac_')[-1].split('_')[0]
                prev_sampled_data_path = f"outputs/{args.resume_checkpoint_path.split('../')[-1].split('_sharded')[0]}.json"
                # prev_sampled_data_path = f"outputs/{args.resume_checkpoint_path.split('../')[-1]}"
                sampled_data = datasets.load_dataset('json', data_files=prev_sampled_data_path, split='train')
                resume = False
            else:
                sharded_model_path = f"{model_path}_sharded/{os.listdir(f'{model_path}_sharded')[0]}/model"
            print("#"*100)
            print("Sampling new data")
            print(f"Steps: {steps}, sampled_data size: {len(sampled_data)}, total data: {al_data_count}")
            print("#"*100)
            if args.sampling_strategy == "cluster":
                with open('/sensei-fs/users/ksaifullah/dolphin_instructions_cluster_sbert.pkl', 'rb') as f:
                    clusters = pickle.load(f)
                random.seed(args.seed)
                sampled_clusters = random.choices(list(clusters.keys()), k=args.num_acquisition_samples)
                shrinked_pool_data = {
                    'id': [],
                    'instruction': [],
                    'input': [],
                    'output': []
                }
                # sampled_data = datasets.load_dataset('json', data_files=args.save_file_name, split='train')
                sampled_data_ids = set(sampled_data['id'])
                for c in tqdm(sampled_clusters):
                    idx = random.sample(range(len(clusters[c]))[:5], 1)[0]
                    sample_id = int(clusters[c][idx][0])  # getting the int from numpy.int64
                    while sample_id in sampled_data_ids:  # we want to get a fresh sample from pool that isn't already present in the training set
                        # sample_cluster = random.sample(list(clusters.keys()), 1)[0]
                        idx = random.sample(range(len(clusters[c]))[:5], 1)[0]
                        sample_id = int(clusters[c][idx][0])
                    shrinked_pool_data['id'].append(pool_data[sample_id]['id'])
                    shrinked_pool_data['instruction'].append(pool_data[sample_id]['instruction'])
                    shrinked_pool_data['input'].append(pool_data[sample_id]['input'])
                    shrinked_pool_data['output'].append(pool_data[sample_id]['output'])

                dataset_w_ppl = Dataset.from_dict(shrinked_pool_data)
            elif args.sampling_strategy == "random":
                print("removing sampled data from pool")
                pool_data.set_format('pandas')
                df = pool_data[:]
                df = df[~df['id'].isin(set(sampled_data['id']))]
                pool_data = Dataset.from_pandas(df, preserve_index=False)
                pool_data = pool_data.shuffle(seed=args.seed)
                shrinked_pool_data = pool_data.select(range(args.num_acquisition_samples))
                dataset_w_ppl = shrinked_pool_data
            else:
                if args.random_pool_fraction:
                    print("removing sampled data from pool")
                    pool_data.set_format('pandas')
                    df = pool_data[:]
                    df = df[~df['id'].isin(set(sampled_data['id']))]
                    pool_data = Dataset.from_pandas(df, preserve_index=False)
                    # ### TEMP ###
                    # shrinked_pool_data = pool_data
                    # ### TEMP ###
                    pool_data = pool_data.shuffle(seed=args.seed)
                    shrinked_pool_data = pool_data.select(range(pool_data_count))
                else:
                    print("Picking pool data from clusters...")
                    with open('/sensei-fs/users/ksaifullah/dolphin_instructions_cluster_sbert.pkl', 'rb') as f:
                        clusters = pickle.load(f)
                    random.seed(args.seed+steps)
                    sampled_clusters = random.choices(list(clusters.keys()), k=pool_data_count)
                    shrinked_pool_data = {
                        'id': [],
                        'instruction': [],
                        'input': [],
                        'output': []
                    }
                    # sampled_data = datasets.load_dataset('json', data_files=args.save_file_name, split='train')
                    sampled_data_ids = set(sampled_data['id'])
                    for c in tqdm(sampled_clusters):
                        idx = random.sample(range(len(clusters[c]))[:5], 1)[0]
                        sample_id = int(clusters[c][idx][0])  # getting the int from numpy.int64
                        while sample_id in sampled_data_ids:  # we want to get a fresh sample from pool that isn't already present in the training set
                            # sample_cluster = random.sample(list(clusters.keys()), 1)[0]
                            idx = random.sample(range(len(clusters[c]))[:5], 1)[0]
                            sample_id = int(clusters[c][idx][0])
                        shrinked_pool_data['id'].append(pool_data[sample_id]['id'])
                        shrinked_pool_data['instruction'].append(pool_data[sample_id]['instruction'])
                        shrinked_pool_data['input'].append(pool_data[sample_id]['input'])
                        shrinked_pool_data['output'].append(pool_data[sample_id]['output'])

                    shrinked_pool_data = Dataset.from_dict(shrinked_pool_data)
                ## preprocess
                eval_preproc = partial(apply_conv_template)
                shrinked_pool_data = shrinked_pool_data.map(eval_preproc)
                print(f"example from shrinked pool data: {shrinked_pool_data['prompt'][0]}")
                num_processes = num_gpus = torch.cuda.device_count()
                try:
                    mp.set_start_method('spawn')  # Required for CUDA in multiprocessing
                except RuntimeError:
                    pass
                result_list = mp.Manager().list()
                processes = []
                for rank in range(num_processes):
                    p = mp.Process(target=inference_worker, args=(rank, sharded_model_path, shrinked_pool_data.shard(num_shards=num_gpus, index=rank), result_list))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
                    p.close()
                print("All processes joined and closed")
                # model = model.cpu()
                # del model
                gc.collect()
                torch.cuda.empty_cache()
                dataset_w_ppl = datasets.concatenate_datasets(result_list)
                # dataset_w_ppl = dataset_w_ppl.sort('ppl')
                dataset_w_ppl = dataset_w_ppl.sort('ppl', reverse=True)
                print(f"Most uncertain example: {dataset_w_ppl['input'][0]}")
                if args.plot_ppl_hist:
                    plt.hist(dataset_w_ppl['ppl'], bins=100)
                    plt.savefig(f"{args.al_data_fraction}_ppl_hist_{steps}.png")
            # we don't want to add more data than the total data count
            if len(sampled_data)+args.num_acquisition_samples <= al_data_count:
                num_acquisition_samples = args.num_acquisition_samples
            else:
                num_acquisition_samples = al_data_count-len(sampled_data)
            if args.stratification_strategy == "mixed":
                # select 70% of the data from the top with high ppl and 30% of the data with low ppl
                top_70 = int(args.mixed_sampling_factor*num_acquisition_samples)
                bottom_30 = num_acquisition_samples - top_70
                top_70_data = dataset_w_ppl.select(range(top_70))
                # bottom_30_data = dataset_w_ppl.select(range(len(dataset_w_ppl)-bottom_30, len(dataset_w_ppl)))
                bottom_30_data = dataset_w_ppl.select(range(top_70, len(dataset_w_ppl))).shuffle(seed=args.seed).select(range(bottom_30))
                acquisition_samples = datasets.concatenate_datasets([top_70_data, bottom_30_data])
            elif args.stratification_strategy == "bucket":
                print("Using bucketing")
                stratified_data = []
                examples_per_bucket = num_acquisition_samples//args.num_k
                total_acquisitions = 0
                for i in range(args.num_k):
                    if args.pick_samples_from == "top":
                        data_shard = dataset_w_ppl.shard(num_shards=args.num_k, index=i, contiguous=True).select(range(examples_per_bucket))
                    elif args.pick_samples_from == "bottom":
                        data_shard = dataset_w_ppl.shard(num_shards=args.num_k, index=i, contiguous=True).select(range(len(dataset_w_ppl)-examples_per_bucket, len(dataset_w_ppl)))
                    elif args.pick_samples_from == "uniform":
                        data_shard = dataset_w_ppl.shard(num_shards=args.num_k, index=i, contiguous=True).shuffle(seed=args.seed).select(range(examples_per_bucket))
                    total_acquisitions += len(data_shard)
                    if i == args.num_k-1 and total_acquisitions < num_acquisition_samples:
                        if args.pick_samples_from == "top":
                            data_shard = dataset_w_ppl.shard(num_shards=args.num_k, index=i, contiguous=True).select(range(examples_per_bucket+num_acquisition_samples-total_acquisitions))
                        elif args.pick_samples_from == "bottom":
                            data_shard = dataset_w_ppl.shard(num_shards=args.num_k, index=i, contiguous=True).select(range(len(dataset_w_ppl)-examples_per_bucket-num_acquisition_samples+total_acquisitions, len(dataset_w_ppl)))
                        elif args.pick_samples_from == "uniform":
                            data_shard = dataset_w_ppl.shard(num_shards=args.num_k, index=i, contiguous=True).shuffle(seed=args.seed).select(range(examples_per_bucket+num_acquisition_samples-total_acquisitions))
                    stratified_data.append(data_shard)
                acquisition_samples = datasets.concatenate_datasets(stratified_data)
            elif args.stratification_strategy == "greedy":
                if args.pick_samples_from == "top":
                    acquisition_samples = dataset_w_ppl.select(range(num_acquisition_samples))
                elif args.pick_samples_from == "bottom":
                    acquisition_samples = dataset_w_ppl.select(range(len(dataset_w_ppl)-num_acquisition_samples, len(dataset_w_ppl)))
                else:
                    print("Choose a valid 'pick_samples_from' option")
                    raise NotImplementedError
            else:
                print("Choose a valid 'stratification_strategy' option")
                raise NotImplementedError
            sampled_data = datasets.concatenate_datasets([sampled_data, acquisition_samples])
            sampled_data.to_json(f"{args.save_file_name}")

            if args.decay_k:
                if args.num_k-1 == 1:
                    args.num_k = 2
                else:
                    args.num_k -= 1
                print(args.num_k)

            steps += 1
            print("#"*100)
            print("Training on new data")
            print(f"Steps: {steps}, sampled_data size: {len(sampled_data)}, total data: {al_data_count}")
            print("#"*100)
            # os.system(f"rm -rf {model_path}_sharded")
            cmd = ["python", "train_AL.py"]
            cmd.extend(["--init_checkpoint_path", f"{args.init_checkpoint_path}", "--model_config_path", f"{args.model_config_path}", "--checkpoint_path", f"{model_path}_sharded", "--wrapped_class_name", "LlamaDecoderLayer", "--data_path", f"{args.save_file_name}", "--hack", "--batch_size", "1", "--accumulation_steps", "8", "--dont_save_opt", "--num_epochs", "2", "--lr", f"{args.lr}", , "--seed", f"{args.seed}", "--filtering_method", "random"]) # , "--wandb", "--wb_name", f"s_{steps}_{model_path.split('/')[-1]}", "--wb_project", "al_data_distillation"
            result = subprocess.run(cmd)

    end_time = time.time()
    print(f"Total time taken: {(end_time-start_time)/60} minutes")
    with open('al_experiments_runtime.txt', 'a+') as f:
        f.write(f"{model_path}: {(end_time-start_time)/60} minutes\n")

    # print("#"*100)
    # print("Converting to HF")
    # print("#"*100)
    # cmd = ["python", "convert_fsdp_to_hf.py"]
    # cmd.extend(["--load_path", f"{model_path}_sharded/{os.listdir(f'{model_path}_sharded')[0]}/model", "--save_path", f"/sensei-fs/users/ksaifullah/{model_path.split('/')[-1]}_hf", "--config_path", args.model_config_path])
    # result = subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint_path", default="/sensei-fs/users/ksaifullah/llama2_7B_sharded", type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--model_config_path", default="/sensei-fs/users/ksaifullah/llama2_7B_hf", type=str)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--template_type", default="alpaca", type=str)
    parser.add_argument("--file_path", default="/sensei-fs/users/ksaifullah/dolphin_1024_ctx.jsonl", type=str)
    parser.add_argument("--save_file_name", default="outputs/al_train_dataset.jsonl", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--cluster_data_fraction", default=0.001, type=float)
    parser.add_argument("--al_data_fraction", default=0.001, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--debug", action='store_true', help="This reduce the number of generation examples to 4, so that we can debug faster.")
    parser.add_argument("--resume", action='store_true', help="This will resume the training from a AL checkpoint.")
    parser.add_argument("--random_pool_fraction", action='store_true', help="Randomly select subset of the data for pool set.")
    parser.add_argument("--resume_checkpoint_path", default=None, type=str, help="AL checkpoint path to resume from.")
    parser.add_argument("--model_ask", action='store_true', help="if active we'll ask the model about instruction.")
    parser.add_argument("--sampling_strategy", default="forward_ppl", type=str, help="Sampling strategy for AL.")
    parser.add_argument("--num_acquisition_samples", default=1000, type=int)
    parser.add_argument("--stratification_strategy", default="greedy", type=str, help="Sampling strategy for AL.")
    parser.add_argument("--pick_samples_from", type=str, choices=["top", "bottom", "uniform"], default="top")
    parser.add_argument("--mixed_sampling_factor", default=0.7, type=float)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--decay_k", action='store_true', help="Decay the number of buckets.")
    parser.add_argument("--model_max_length", default=1024, type=int, help="Model max length.")
    parser.add_argument("--plot_ppl_hist", action='store_true', help="Plot the histogram of ppl.")
    args = parser.parse_args()
    print(args)
    main(args)