"""This script generates the evaluation responses that can e used by eval_scoring.py"""
import argparse
from functools import partial

from datasets import Dataset
import transformers
import torch

from io_utils import load_jsonlines
from utils import load_fsdp_ckpt_with_accelerate
from conversation import get_conv_template

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def generate_responses_batched(example, model, tokenizer, template_type, kwargs):
    # preprocess instructions into prompted inputs
    prompts = []
    for instruction in example['instruction']:
        conv = get_conv_template(template_type)
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], "")
        prompts.append( conv.get_prompt())
    encoding = tokenizer(prompts, 
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

    example.update({"text": decoded_output})
    example.update({"metadata": [kwargs] * len(decoded_output)})

    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/fs/nexus-scratch/pchiang/llama/7B_sharded", type=str)
    parser.add_argument("--dummy_path", default="/fs/nexus-scratch/pchiang/llama/7B_hf", type=str)
    parser.add_argument("--template_type", default="alpaca", type=str)
    parser.add_argument("--file_path", default="datasets/self-instruct-val(processed).jsonl", type=str)
    parser.add_argument("--save_file_name", default="outputs/answers/self-instruct_llama7B.jsonl", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--debug", action='store_true', help="This reduce the number of generation examples to 4, so that we can debug faster.")
    args = parser.parse_args()

    model_config = transformers.AutoConfig.from_pretrained(args.dummy_path)
    if "llama" in args.model:
        model_config.vocab_size += 1 # hardcode the vocab size for llama... 

    model = load_fsdp_ckpt_with_accelerate(args.model, model_config, hf_dummy_path=args.dummy_path, wrapped_class="LlamaDecoderLayer" if 'llama' in args.model else "OPTDecoderLayer")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.dummy_path,
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
    tokenizer.add_special_tokens(special_tokens_dict)
    
    ## set the models to eval mode
    model = model.eval()
    
    file_path = args.file_path
    tasks = load_jsonlines(args.file_path)
    # ### debug
    if args.debug:
        tasks = tasks[:4]


    raw_data = Dataset.from_list(tasks)
    ## rename columns for dolly eval
    if "dolly" in args.file_path:
        question_sources = "dolly"
    elif "vicuna" in args.file_path:
        question_sources = "vicuna"
        raw_data = raw_data.rename_column("text", "instruction")
    elif "user_oriented_instructions" in args.file_path:
        question_sources = "alpaca"

    ## run generation
    generate_kwargs = dict(max_length=1024, do_sample=False, top_p=0.1, 
                           num_return_sequences=1, temperature=0.1, 
                           repetition_penalty=1.2)
    generate = partial(generate_responses_batched, 
                       model=model,  
                       tokenizer=tokenizer,
                       template_type=args.template_type,
                       kwargs=generate_kwargs)

    dataset_w_responses = raw_data.map(generate,
                                            batched=True,
                                            batch_size=args.batch_size)

    dataset_w_responses.to_json(args.save_file_name)



