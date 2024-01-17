from datasets import concatenate_datasets
from datasets import load_dataset
import os
import json
import io
from tqdm import tqdm


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

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


files = os.listdir('outputs')
files = [file for file in files if file.endswith('.jsonl')]
# sorting file names based on a number in the file name
files = sorted(files, key=lambda x: int(x.split('-')[0].split('_')[-1]))
datasets = [load_dataset('json', data_files=f"outputs/{file}", split='train') for file in files]

# concatenate all datasets
dataset = concatenate_datasets(datasets)

org_data_path = "datasets/alpaca-train.jsonl"
org_dataset = jload(org_data_path)

for i in tqdm(range(len(org_dataset))):
    if org_dataset[i]['instruction'] != dataset[i]['instruction']:
        print(i)
        print(org_dataset[i]['instruction'])
        print(dataset[i]['instruction'])
        print('------------------')
    else:
        org_dataset[i]['output'] = dataset[i]['output']

jdump(org_dataset, "./datasets/llama_chat_data.jsonl")
# from IPython import embed; embed()

