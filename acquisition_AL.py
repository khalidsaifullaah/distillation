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

from io_utils import load_jsonlines
from utils import load_fsdp_ckpt_with_accelerate, add_padding_token
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
    source  = prompt_input.format_map(example) if example.get("instruction", "") != "" else prompt_no_input.format_map(example)
    # source = confidence_prompt.format_map(example)
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

def compute_perplexity_batched(example, model, tokenizer, device=0, kwargs=None):
    prompt = example['prompt']
    encoding = tokenizer(prompt, 
                          return_tensors="pt",
                          padding="longest",
                          max_length=tokenizer.model_max_length,
                          truncation=True,
                      )
    encoding.pop('token_type_ids', None)
    encoding = {k: v.to(f"cuda:{device}") for k,v in encoding.items()}
    with torch.no_grad():
        model_output = model(**encoding)
        if torch.isnan(model_output.logits).any():
            print("NAN")
        loss = get_clm_loss(encoding['input_ids'], model_output.logits.float())
        # kwargs['max_new_tokens'] = 3
        # model_output = model.generate(**encoding, **kwargs)
        # logits = torch.nan_to_num(torch.stack(model_output.scores).transpose(0,1))
        # token_probs = torch.nn.functional.softmax(logits, dim=-1)
        # confidence = token_probs[:, :, torch.tensor([1939,694])].view(len(prompt), -1).mean(dim=-1).tolist() # 1939: No, 694: no
        # labels = model_output.sequences[:, encoding['input_ids'].shape[-1]:]
        # ppl = loss = get_clm_loss(labels, logits.float()) # using log perplexity
        log_ppl = loss.tolist()
        # input_len = encoding['input_ids'].shape[-1]
        # output_sequences = model_output.sequences[:, input_len:].cpu()
        # decoded_output = tokenizer.batch_decode(output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    del example['prompt']
    # example.update({"ppl": log_ppl})
    # example.update({"decoded_output": decoded_output})
    # return example
    return log_ppl
    # return confidence

def inference_worker(rank, sharded_model_path, data_partition, result_list):
    gpu = rank
    model_config = transformers.AutoConfig.from_pretrained(args.model_config_path)
    model = load_fsdp_ckpt_with_accelerate(sharded_model_path, model_config, hf_dummy_path=args.model_config_path, wrapped_class="LlamaDecoderLayer", rank=gpu)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_config_path,
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
    model.eval()
    generate_kwargs = dict(max_new_tokens=256, do_sample=False, temperature=0.6,
        num_return_sequences=1, output_scores=True, return_dict_in_generate=True, 
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]))
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
    pool_data = pool_data.add_column('id', list(range(len(pool_data))))
    sampled_data = pool_data.shuffle(seed=args.seed).select(range(1000))

    pool_data_count = int(len(pool_data)*args.cluster_data_fraction)
    al_data_count = int(len(pool_data)*args.al_data_fraction)

    model_path = f"/home/ksaifullah/al_dolphin_llama2_7B_dfrac_{args.al_data_fraction}_poolfrac_{args.cluster_data_fraction}_forward_ppl"
    # model_path = f"/home/ksaifullah/al_dolphin_llama2_7B_dfrac_{args.al_data_fraction}_poolfrac_{args.cluster_data_fraction}_self_conf"
    initial_run = True
    resume = args.resume
    if resume:
        initial_run = False
    steps = 0
    start_time = time.time()
    while len(sampled_data) < al_data_count:
        if initial_run:
            # saving sampled data
            sampled_data.to_json(f"{args.save_file_name.split('.')[0]}_{args.al_data_fraction}.json")
            print("#"*100)
            print("Running seed training")
            print(f"steps: {steps}, sampled_data size: {len(sampled_data)}, total data: {al_data_count}")
            print("#"*100)
            cmd = ["python", "train_AL.py"]
            cmd.extend(["--init_checkpoint_path", "/home/ksaifullah/llama2_7B_sharded", "--model_config_path", "/home/ksaifullah/llama2_7B_hf", "--checkpoint_path", f"{model_path}_sharded", "--wrapped_class_name", "LlamaDecoderLayer", "--data_path", args.save_file_name, "--hack", "--batch_size", "1", "--accumulation_steps", "8", "--dont_save_opt", "--wandb", "--wb_name", f"s_{steps}_{model_path.split('/')[-1]}", "--wb_project", "al_data_distillation", "--num_epochs", "2"])
            result = subprocess.run(cmd)
            initial_run = False

        else:
            if resume:
                sharded_model_path = f"{args.resume_checkpoint_path}/{os.listdir(args.resume_checkpoint_path)[0]}/model"
                prev_data_fraction = args.resume_checkpoint_path.split('dfrac_')[-1].split('_')[0]
                prev_sampled_data_path = f"{args.save_file_name.split('.')[0]}_{prev_data_fraction}.json"
                sampled_data = datasets.load_dataset('json', data_files=prev_sampled_data_path, split='train')
                resume = False
            else:
                sharded_model_path = f"{model_path}_sharded/{os.listdir(f'{model_path}_sharded')[0]}/model"
            print("#"*100)
            print("Sampling new data")
            print(f"Steps: {steps}, sampled_data size: {len(sampled_data)}, total data: {al_data_count}")
            print("#"*100)
            # model_config = transformers.AutoConfig.from_pretrained(args.model_config_path)
            # if isinstance(model_config, LlamaConfig):
            #     model_config.vocab_size += 1 # hardcode the vocab size for llama...
            # model = load_fsdp_ckpt_with_accelerate(args.model, model_config, hf_dummy_path=args.model_config_path, wrapped_class="LlamaDecoderLayer" if 'llama' in args.model else "OPTDecoderLayer", rank=0)
            # # cmd = ["python", "convert_fsdp_to_hf.py"]
            # # cmd.extend(["--load_path", f"{model_path}_sharded/{os.listdir(f'{model_path}_sharded')[0]}/model", "--save_path", f"{model_path}_hf", "--config_path", args.model_config_path])
            # # result = subprocess.run(cmd)

            # # model = transformers.AutoModelForCausalLM.from_pretrained(f"{model_path}_hf", torch_dtype=torch.bfloat16, trust_remote_code=True)
            # tokenizer = transformers.AutoTokenizer.from_pretrained(
            #         f"{model_path}_hf",
            #         model_max_length=2048,
            #         padding_side="left",
            #         use_fast=True,
            #     )
            # add_padding_token(tokenizer)

            #remove the sampled data from the pool data with .filter()
            # pool_data.set_format('pandas')
            # df = pool_data[:]
            # df = df[~df.index.isin(sampled_data['id'])]
            # pool_data = Dataset.from_pandas(df, preserve_index=False)

            with open('clusters.pkl', 'rb') as f:
                clusters = pickle.load(f)
            random.seed(args.seed)
            sampled_clusters = random.choices(list(clusters.keys()), k=pool_data_count)
            cluster_sampling_data = {
                'id': [],
                'instruction': [],
                'input': [],
                'output': []
            }
            from IPython import embed; embed()
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
            num_processes = num_gpus = torch.cuda.device_count()
            try:
                mp.set_start_method('spawn')  # Required for CUDA in multiprocessing
            except RuntimeError:
                pass
            result_list = mp.Manager().list()
            processes = []
            for rank in range(num_processes):
                p = mp.Process(target=inference_worker, args=(rank, sharded_model_path, cluster_sampling_data.shard(num_shards=num_gpus, index=rank), result_list))
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
            dataset_w_ppl = dataset_w_ppl.sort('ppl', reverse=True)
            # count the frequency of ppl in the dataset
            # from collections import Counter
            # ppl_freq = Counter(dataset_w_ppl['ppl'])
            # we don't want to add more data than the total data count
            if len(sampled_data)+1000 <= al_data_count:
                acquisition_samples = dataset_w_ppl.select(range(1000))
            else:
                acquisition_samples = dataset_w_ppl.select(range(al_data_count-len(sampled_data)))
            acquisition_samples = acquisition_samples.remove_columns('ppl')
            sampled_data = datasets.concatenate_datasets([sampled_data, acquisition_samples])
            sampled_data.to_json(f"{args.save_file_name.split('.')[0]}_{args.al_data_fraction}.json")

            steps += 1
            print("#"*100)
            print("Training on new data")
            print(f"Steps: {steps}, sampled_data size: {len(sampled_data)}, total data: {al_data_count}")
            print("#"*100)
            os.system(f"rm -rf {model_path}_sharded")
            cmd = ["python", "train_AL.py"]
            cmd.extend(["--init_checkpoint_path", "/home/ksaifullah/llama2_7B_sharded", "--model_config_path", "/home/ksaifullah/llama2_7B_hf", "--checkpoint_path", f"{model_path}_sharded", "--wrapped_class_name", "LlamaDecoderLayer", "--data_path", f"{args.save_file_name.split('.')[0]}_{args.al_data_fraction}.json", "--hack", "--batch_size", "1", "--accumulation_steps", "8", "--dont_save_opt", "--wandb", "--wb_name", f"s_{steps}_{model_path.split('/')[-1]}", "--wb_project", "al_data_distillation", "--num_epochs", "2"])
            result = subprocess.run(cmd)

    end_time = time.time()
    print(f"Total time taken: {(end_time-start_time)/60} minutes")
    with open('al_experiments_runtime.txt', 'a+') as f:
        f.write(f"{model_path}: {(end_time-start_time)/60} minutes\n")

    print("#"*100)
    print("Converting to HF")
    print("#"*100)
    cmd = ["python", "convert_fsdp_to_hf.py"]
    cmd.extend(["--load_path", f"{model_path}_sharded/{os.listdir(f'{model_path}_sharded')[0]}/model", "--save_path", f"/sensei-fs/users/ksaifullah/{model_path}_hf", "--config_path", args.model_config_path])
    result = subprocess.run(cmd)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ksaifullah/llama2_7B_sharded", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--model_config_path", default="/home/ksaifullah/llama2_7B_hf", type=str)
    parser.add_argument("--template_type", default="alpaca", type=str)
    parser.add_argument("--file_path", default="datasets/dolphin.jsonl", type=str)
    parser.add_argument("--save_file_name", default="outputs/al_train_dataset.jsonl", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--cluster_data_fraction", default=0.001, type=float)
    parser.add_argument("--al_data_fraction", default=0.001, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--debug", action='store_true', help="This reduce the number of generation examples to 4, so that we can debug faster.")
    parser.add_argument("--resume", action='store_true', help="This will resume the training from a AL checkpoint.")
    parser.add_argument("--resume_checkpoint_path", default=None, type=str, help="AL checkpoint path to resume from.")
    args = parser.parse_args()
    main(args)