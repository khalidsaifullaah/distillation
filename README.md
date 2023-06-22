# QuickStart
First, you want to install the environment (assuming that you have conda installed)

`conda env create -f environment.yml`

To finetune 7B llama on the self instruct dataset, run the command below

```python train.py --init_checkpoint_path /fs/nexus-scratch/pchiang/llama/7B_sharded --checkpoint_path $YOUR_CHECKPOINT_PATH```

I have preconverted llama 7B fsdp format and saved it at `/fs/nexus-scratch/pchiang/llama/7B_sharded` I found this to be necessary to reduce the memory footprint during the loading phase.

# Converting checkpoints from huggingface to fsdp

The `convert_hf_to_fsdp.py` converts huggingface checkpoint to one that can be loaded by fsdp in a distributed. After conversion, the model can be loaded in a distributed manner without consumming too much memory. Usually, when loading the hugging face model to N gpus, one needs to first realize N models in cpu memory before moving model to gpus. This is can easily blow out the CPU memory if the model is large. You can convert the model by running the command below

```python convert_hf_to_fsdp.py --load_path $HF_CHECKPOINT_PATH --save_path $SAVE_PATH ```

# Other things
There are still many bugs and kinks that have to be worked out in the repository. Some of the things that come into mind are

1. The modification of architecture has to happen before sharding which is quite inconvenient.
2. the environment file includes many unnecessary things and can probably be improved.

Feel free to initiate a pull request. I will continue to improve the repo as things move along.

# Memory Usage on 4 gpus
| Model         | Actual Max GPU Memory | Theoretical Max GPU Memory (Theoretical) |
| ------------- | ---------- | ----------- | 
| Llama 7B bf16 (13GB) | 4.3 GB    |  3.2 GB    |
| Llama 7B bf16 (13GB) 1000 tokens no grad forward | 4.9 GB    |  ?? GB    | 
| Llama 7B bf16 (13GB) 1000 tokens grad forward | 19.8 GB    |  3.2 GB (Parameters) + 3.2 GB (Gradients) + 9.2 GB (Activations) =   | 


