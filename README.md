# Sample efficient instruction tuning for LLMs
When we have a large pool of unlabeled instruction data available, how can we determine the best instructions to provide to human annotators? One approach is to set up an active learning framework for this task. With abundant unlabeled data, active learning allows us to iteratively select the most informative examples to label. By starting with a small labeled set and iteratively growing it with strategically chosen examples, we can find highly effective instructions much more efficiently than labeling randomly selected data.

# QuickStart
First, you want to install the environment (assuming that you have conda installed)

`conda env create -f environment.yml`

To start an experiment of distilling dataset (for 500, 1000, 1500, 2000, 2500, 3000, and 4000 samples respectively) w/ llama 7B, run the command below:

```shell
### Stratified bucket (top) ###
# Define common parameters
INIT_CHECKPOINT_PATH="/path/of/model/llama2_7B_sharded"
MODEL_CONFIG_PATH="/path/of/model/llama2_7B_hf"
DATA_PATH="/path/of/model/databricks-dolly-15k.jsonl"
LR="--lr 5e-5"
BATCH_SIZE="--batch_size 4"
AL_DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000")
PREV_AL_FRACTION=""  # Initialize to an empty string

for al_fraction in "${AL_DATA_FRACTIONS[@]}"
do
  ACQUISITION_MODEL_PATH="/home/ksaifullah/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_top_forward_ppl"
  EVAL_MODEL_PATH="../al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_top_forward_ppl_sharded"
  EVAL_SAVE_FILE_NAME="/path/of/model/dolly_llama2_7B_outputs/stratify_top/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_top_forward_ppl_seed42.json"
  # Check if it's not the first iteration
  if [ -n "$PREV_AL_FRACTION" ]
  then
    RESUME="--resume"
    RESUME_CHECKPOINT_PATH="../al_dolly_llama2_7B_numdata_${PREV_AL_FRACTION}_bucket_stratify_5_top_forward_ppl_sharded"
  else
    RESUME=""
    RESUME_CHECKPOINT_PATH=""  # Leave it empty for the first iteration
  fi
  
  # Aquiring samples from the unlabelled pool and training the model
  python acquisition_AL.py \
    --file_path "$DATA_PATH" \
    --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
    --model_config_path "$MODEL_CONFIG_PATH" \
    $BATCH_SIZE \
    --al_data_fraction "$al_fraction" \
    --cluster_data_fraction 1.00 \
    $LR \
    --num_acquisition_samples 100 \
    --random_pool_fraction \
    --stratification_strategy bucket \
    --model_path "$ACQUISITION_MODEL_PATH" \
    --seed 42 \
    --num_k 5 \
    --pick_samples_from top \
    $RESUME \
    $([ -n "$RESUME_CHECKPOINT_PATH" ] && echo "--resume_checkpoint_path '$RESUME_CHECKPOINT_PATH'")

  # Evaluate the model on Alpaca Eval dataset
  python eval_generate.py \
    --sharded_model "$EVAL_MODEL_PATH" \
    --model_config_path "$MODEL_CONFIG_PATH" \
    --file_path alpaca_eval \
    --save_file_name "$EVAL_SAVE_FILE_NAME"
done
```

You need to convert llama 7B in fsdp format, It's necessary to reduce the memory footprint during the model loading phase.

# Converting checkpoints from huggingface to fsdp

The `convert_hf_to_fsdp.py` converts huggingface checkpoint to one that can be loaded by fsdp in a distributed. After conversion, the model can be loaded in a distributed manner without consumming too much memory. Usually, when loading the hugging face model to N gpus, one needs to first realize N models in cpu memory before moving model to gpus. This is can easily blow out the CPU memory if the model is large. You can convert the model by running the command below

```python convert_hf_to_fsdp.py --load_path $HF_CHECKPOINT_PATH --save_path $SAVE_PATH ```

Similarly, you can use the `convert_fsdp_to_hf.py` script to convert fsdp checkpoints back to hf format.

Feel free to initiate a pull request. I will continue to improve the repo as things move along.

# Memory Usage on 4 gpus
| Model         | Actual Max GPU Memory | Theoretical Max GPU Memory (Theoretical) |
| ------------- | ---------- | ----------- | 
| Llama 7B bf16 (13GB) | 4.3 GB    |  3.2 GB    |
| Llama 7B bf16 (13GB) 1000 tokens no grad forward | 4.9 GB    |  ?? GB    | 
| Llama 7B bf16 (13GB) 1000 tokens grad forward | 19.8 GB    |  3.2 GB (Parameters) + 3.2 GB (Gradients) + 9.2 GB (Activations) =   | 


