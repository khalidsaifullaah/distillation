#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import io
import json
import copy
import logging
import random
from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import pickle

import torch
import transformers
from torch.utils.data import Dataset
from datasets import load_dataset
from datasets import Dataset as HFDataset

IGNORE_INDEX = -100
PROMPT_DICT = {
    "dolphin": {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{input}\n\n### Response:"
        ),},
    "alpaca": {    
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )},

}

column_mappting = {
    'instruction': 'instruction',
    'Instruction': 'instruction',
    'input': 'input',
    'Input': 'input',
    'context': 'input',
    'output': 'output',
    'Output': 'output',
    'response': 'output'
}

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)
from io_utils import read_jsonlines
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_fraction: float=1.0, seed: int=42, efficient_load: bool=False, filtering_method: str='random'):
        super().__init__()
        logging.warning("Loading data...")
        if efficient_load:
            data_dict = load_dataset('json', data_files=data_path, split='train')
            if data_fraction > 1:
                used_data_count = int(data_fraction)
            else:
                used_data_count = int(len(data_dict)*data_fraction)
            if filtering_method == 'random':
                data_dict = data_dict.shuffle(seed=seed).select(range(used_data_count))
            elif filtering_method == 'no_shuffle':
                data_dict = data_dict.select(range(used_data_count))
            elif filtering_method == 'cluster':
                print("filtering data based on clusters")
                with open('/sensei-fs/users/ksaifullah/dolphin_instructions_cluster_sbert.pkl', 'rb') as f:
                    clusters = pickle.load(f)
                random.seed(seed)
                sampled_clusters = random.choices(list(clusters.keys()), k=used_data_count)
                filtered_data = {
                    'instruction': [],
                    'input': [],
                    'output': []
                }
                for c in sampled_clusters:
                    idx = random.sample(range(len(clusters[c]))[:3], 1)[0]
                    sample_id = int(clusters[c][idx][0])  # getting the int from numpy.int64
                    filtered_data['instruction'].append(data_dict[sample_id]['instruction'])
                    filtered_data['input'].append(data_dict[sample_id]['input'])
                    filtered_data['output'].append(data_dict[sample_id]['output'])
                data_dict = HFDataset.from_dict(filtered_data)
                data_dict.to_json("train_data.jsonl")
            else:
                raise ValueError(f"Unexpected filtering method: {filtering_method}, choose from ['random', 'cluster']")
            print(f"using {used_data_count} data out of {len(data_dict)}")
            columns = data_dict.column_names
            # changing column names
            print(f"changing column names from {columns[:3]} to {[column_mappting[c] for c in columns[:3]]}")
            if columns[0] != column_mappting[columns[0]]:
                data_dict = data_dict.rename_column(columns[0], column_mappting[columns[0]])
            if columns[1] != column_mappting[columns[1]]:
                data_dict = data_dict.rename_column(columns[1], column_mappting[columns[1]])
            if columns[2] != column_mappting[columns[2]]:
                data_dict = data_dict.rename_column(columns[2], column_mappting[columns[2]])
            logging.warning("Formatting inputs...")
            data_format = "alpaca" if "dolly" in data_path or "alpaca" in data_path else "dolphin"
            prompt_input, prompt_no_input = PROMPT_DICT[data_format]["prompt_input"], PROMPT_DICT[data_format]["prompt_no_input"]
            def _format_data(examples):
                output = []
                for row in zip(examples["instruction"], examples["input"], examples["output"]):
                    examples = {"instruction": row[0], "input": row[1], "output": row[2]}
                    context_col_name = "input" if "dolly" in data_path or "alpaca" in data_path else "instruction"
                    if examples.get(context_col_name, "") != "":
                        output += [prompt_input.format_map(examples)]
                    else:
                        output += [prompt_no_input.format_map(examples)]
                return {"source": output}
            self.sources = data_dict.map(_format_data, remove_columns=data_dict.column_names, batched=True)['source']
            self.targets = data_dict.map(lambda examples: {"target": [f"{example}{tokenizer.eos_token}" for example in examples['output']]}, remove_columns=data_dict.column_names, batched=True)['target']
            return
        if "dolly" in data_path:
            list_data_dict = read_jsonlines(data_path)
            list_data_dict = list(list_data_dict)
            list_data_dict = [{"instruction": data_dict["instruction"],
                                "input": data_dict["context"],
                                "output": data_dict["response"]} for data_dict in list_data_dict]
        else:
            list_data_dict = jload(data_path)
        used_data_count = int(len(list_data_dict)*data_fraction)
        print(f"using {used_data_count} data out of {len(list_data_dict)}")
        random.seed(seed)
        random.shuffle(list_data_dict)
        list_data_dict = list_data_dict[:used_data_count]

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        # return len(self.input_ids)
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        return dict(sources=self.sources[i], targets=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            sources.append(instance['sources'])
            targets.append(instance['targets'])
        instances = preprocess(sources, targets, self.tokenizer)
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids, labels = tuple((instances['input_ids'], instances['labels']))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path, data_fraction: float=1.0, seed: int=42, efficient_load: bool=False, filtering_method: str='random') -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, data_fraction=data_fraction, seed=seed, efficient_load=efficient_load, filtering_method=filtering_method)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)