#!/bin/bash

#SBATCH --partition scavenger

#SBATCH --ntasks=4

#SBATCH --mem=32G

#SBATCH --account=scavenger

#SBATCH --qos=scavenger

#SBATCH --time=12:00:00

#SBATCH --output=logs/eval-metrics-%A.out

#SBATCH --job-name=eval

source ~/.bashrc
conda activate alpaca


full_model='/fs/cml-projects/instruction_following/alpaca-llama7B_self-instruct_dataf1.0/1935/model'
dummy_path='/fs/nexus-scratch/pchiang/llama/7B_hf'
# full_model='/fs/cml-projects/instruction_following/alpaca-opt-6.7b_dataf1.0/2030/model'
# dummy_path='/fs/cml-projects/instruction_following/pretrained_models/opt6.7b_hf'
output_dir=data_instruct/vicuna-eval

for frac in 0.2 0.4 0.6 0.8;
do
python ./eval_generate.py \
    --batch_size 8 \
    --dummy_path ${dummy_path} \
    --model1 ${full_model} \
    --model2 /fs/cml-projects/instruction_following/alpaca-llama7B_self-instruct_dataf${frac}/1935/model \
    --file_path 'datasets/user_oriented_instructions.jsonl' \
    --save_file_name data_instruct/eval_responses-llama7b-pre-${frac}-self_instruct.jsonl; \
python ./model_eval_vicuna.py \
    -q data_instruct/eval_responses-llama7b-pre-${frac}-self_instruct.jsonl \
    -p vicuna_prompts/prompt.jsonl \
    -r vicuna_prompts/reviewer.jsonl \
    -o ${output_dir}/llama7b-pre-${frac}-self_instruct.jsonl;
done