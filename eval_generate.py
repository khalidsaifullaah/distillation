"""This script generates the evaluation responses that can e used by eval_scoring.py"""
import argparse
from functools import partial
import json
from typing import Dict

from datasets import Dataset
import datasets
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.models.llama.configuration_llama import LlamaConfig
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from io_utils import load_jsonlines
from utils import load_fsdp_ckpt_with_accelerate
from conversation import get_conv_template

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # stop when [EOS] token is generated
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

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

def generate_responses_batched(example, model, tokenizer, kwargs):
    prompt = example['prompt']
    encoding = tokenizer(prompt, 
                          return_tensors="pt",
                          padding="longest",
                          max_length=tokenizer.model_max_length,
                          truncation=True,
                      )
    encoding = encoding.to(model.device)
    with torch.no_grad():
        model_output = model.generate(**encoding, **kwargs)
        input_len = encoding.input_ids.shape[-1]
        model_output = model_output[:, input_len:].cpu()
        decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    del example['prompt']
    example.update({"output": decoded_output})
    kwargs.pop("stopping_criteria", None)
    example.update({"metadata": [kwargs] * len(decoded_output)})

    return example

def inference_worker(rank, model, tokenizer, data_partition, result_list):
    gpu = rank
    device = torch.device(f'cuda:{gpu}')
    model = model.to(device)
    model.eval()
    generate_kwargs = dict(max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9,
        num_return_sequences=1, stopping_criteria=StoppingCriteriaList([StopOnTokens()]))
    get_ppl = partial(generate_responses_batched, 
                    model=model,  
                    tokenizer=tokenizer,
                    kwargs=generate_kwargs)

    dataset_w_responses = data_partition.map(get_ppl, batched=True, batch_size=args.batch_size)
    result_list.append(dataset_w_responses)
    # model = model.cpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/fs/nexus-scratch/pchiang/llama/7B_sharded", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--model_config_path", default="/fs/nexus-scratch/pchiang/llama/7B_hf", type=str)
    parser.add_argument("--template_type", default="alpaca", type=str)
    parser.add_argument("--file_path", default="datasets/self-instruct-val(processed).jsonl", type=str)
    parser.add_argument("--save_file_name", default="outputs/answers/self-instruct_llama7B.jsonl", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--debug", action='store_true', help="This reduce the number of generation examples to 4, so that we can debug faster.")
    args = parser.parse_args()

    model_config = transformers.AutoConfig.from_pretrained(args.model_config_path)
    if isinstance(model_config, LlamaConfig):
        model_config.vocab_size += 1 # hardcode the vocab size for llama... 
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_config_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    # model = load_fsdp_ckpt_with_accelerate(args.model, model_config, hf_dummy_path=args.model_config_path, wrapped_class="LlamaDecoderLayer" if 'llama' in args.model else "OPTDecoderLayer")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_config_path,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,
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
    
    
    if "dolly" in args.file_path or "vicuna" in args.file_path or "user_oriented_instructions" in args.file_path:
        tasks = load_jsonlines(args.file_path)
        raw_data = Dataset.from_list(tasks)
        if "dolly" in args.file_path:
            question_sources = "dolly"
        elif "vicuna" in args.file_path:
            question_sources = "vicuna"
            raw_data = raw_data.rename_column("text", "instruction")
        elif "user_oriented_instructions" in args.file_path:
            question_sources = "alpaca"
    elif "alpaca_eval" in args.file_path:
        raw_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
        # set generator field to model name
        raw_data = raw_data.map(lambda x: {"generator": args.model_name if args.model_name else args.model_config_path})

    # reduce number of examples for debugging
    if args.debug:
        raw_data = raw_data.select(range(4))

    ## preprocess
    eval_preproc = partial(apply_conv_template, template_type=args.template_type)
    raw_data = raw_data.map(eval_preproc)

    num_processes = num_gpus = torch.cuda.device_count()
    try:
        mp.set_start_method('spawn')  # Required for CUDA in multiprocessing
    except RuntimeError:
        pass
    result_list = mp.Manager().list()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=inference_worker, args=(rank, model, tokenizer, raw_data.shard(num_shards=num_gpus, index=rank), result_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        p.close()

    dataset_w_responses = datasets.concatenate_datasets(result_list)
    if "alpaca_eval" in args.file_path:
        # stringify the metadata, so that it becomes hashable
        dataset_w_responses = dataset_w_responses.map(lambda x: {"metadata": json.dumps(x["metadata"])})
        dataset_w_responses.to_json(args.save_file_name, orient="records", lines=False, indent=True)
    else:
        dataset_w_responses.to_json(args.save_file_name)