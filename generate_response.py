"""This script generates the evaluation responses that can e used by eval_scoring.py"""
import argparse
from functools import partial

from datasets import Dataset, load_dataset
import transformers
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

from io_utils import load_jsonlines
from utils import load_fsdp_ckpt_with_accelerate
from conversation import get_conv_template

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
        stop_ids = [0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def generate_responses_batched(example, model, tokenizer, template_type, kwargs):
    # preprocess instructions into prompted inputs
    prompts = []
    from IPython import embed; embed()
    for instruction in example['instruction']:
        conv = get_conv_template(template_type)
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], "")
        prompts.append(conv.get_prompt())
    encoding = tokenizer(prompts,
                          return_tensors="pt",
                          padding="longest",
                          max_length=tokenizer.model_max_length,
                          truncation=True,
                      )
    encoding = encoding.cuda()
    with torch.no_grad():
        model_output = model.generate(**encoding, **kwargs)
        input_len = encoding.input_ids.shape[-1]
        model_output = model_output[:, input_len:].cpu()
        decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    example.update({"text": decoded_output})
    example.update({"metadata": [kwargs] * len(decoded_output)})

    return example

def generate_response_non_batched(example, model, tokenizer, template_type, kwargs):
    # from IPython import embed; embed()
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    source  = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    # conv = get_conv_template(template_type)
    # conv.append_message(conv.roles[0], example['instruction'])
    # conv.append_message(conv.roles[1], "")
    # prompt = conv.get_prompt()
    encoding = tokenizer(source,
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
        # decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decoded_output = tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # from IPython import embed; embed()
    # example.update({"text": decoded_output})
    # example.update({"metadata": [kwargs] * len(decoded_output)})
    example['output'] = decoded_output
    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded", type=str)
    parser.add_argument("--hf_path", default="/cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf", type=str)
    parser.add_argument("--template_type", default="alpaca", type=str)
    parser.add_argument("--wrapped_class", default="MPTBlock", type=str)
    parser.add_argument("--file_path", default="datasets/alpaca-train.jsonl", type=str)
    parser.add_argument("--save_file_name", default="outputs/MPT_7B_chat_alpaca.jsonl", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--debug", action='store_true', help="This reduce the number of generation examples to 4, so that we can debug faster.")
    parser.add_argument("--parallelize", action='store_true', help="Helps you run multiple jobs in parallel on different chunks of data.")
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=10000, type=int)
    args = parser.parse_args()

    # model_config = transformers.AutoConfig.from_pretrained(args.dummy_path, trust_remote_code=True)
    # if "llama" in args.model:
    #     model_config.vocab_size += 1 # hardcode the vocab size for llama...

    config = transformers.AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    config.vocab_size += 1
    with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    model.tie_weights()
    model = load_checkpoint_and_dispatch(
        model, args.hf_path, device_map="auto", dtype="float16", no_split_module_classes=["MPTBlock"]
    )

    # model = load_fsdp_ckpt_with_accelerate(args.model, model_config, hf_dummy_path=args.dummy_path, wrapped_class=args.wrapped_class)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.hf_path,
            model_max_length=config.max_seq_len,
            padding_side="left",
            # use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = tokenizer.eos_token
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
    # loading .jsonl file with HF datasets
    dataset = load_dataset("json", data_files=file_path)
    # remove the output column
    dataset = dataset.remove_columns(["output"])
    # ### debug
    if args.debug:
        # dataset = dataset[:4]
        # selecting only the first 4 examples
        raw_data = dataset['train'].select(range(4))
    else:
        if args.parallelize:
            raw_data = dataset['train'].select(range(args.start_idx, args.end_idx))
        else:
            raw_data = dataset['train']
    ## rename columns for dolly eval
    if "dolly" in args.file_path:
        question_sources = "dolly"
    elif "vicuna" in args.file_path:
        question_sources = "vicuna"
        raw_data = raw_data.rename_column("text", "instruction")
    elif "user_oriented_instructions" in args.file_path:
        question_sources = "alpaca"

    ## run generation
    generate_kwargs = dict(max_length=1024, do_sample=True, top_p=0.95, top_k=50,
                           num_return_sequences=1, temperature=1.0, 
                           stopping_criteria=StoppingCriteriaList([StopOnTokens()]))
    # generate = partial(generate_responses_batched, 
    #                    model=model,  
    #                    tokenizer=tokenizer,
    #                    template_type=args.template_type,
    #                    kwargs=generate_kwargs)
    # dataset_w_responses = raw_data.map(generate,
    #                                    batched=True,
    #                                    batch_size=args.batch_size, 
    #                                    writer_batch_size=args.batch_size, num_proc=1)
    generate = partial(generate_response_non_batched, 
                       model=model,  
                       tokenizer=tokenizer,
                       template_type=args.template_type,
                       kwargs=generate_kwargs)
    dataset_w_responses = raw_data.map(generate)
    file_path, ext = args.save_file_name.split('.')
    dataset_w_responses.to_json(file_path+f"_{args.start_idx}-{args.end_idx}."+ext)