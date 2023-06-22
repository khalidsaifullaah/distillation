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


# full_model='/fs/cml-projects/instruction_following/alpaca-llama7B_self-instruct_dataf0.4/1935/model'
full_model='/fs/cml-projects/instruction_following/alpaca-opt-1.3b_1.0'

for data_fraction in '0.2';
do 
    python ./eval_generate.py \
    --dummy_path "/fs/nexus-scratch/pchiang/llama/7B_hf" \
    --model1 ${full_model} \
    --model2 "/fs/cml-projects/instruction_following/alpaca-opt-1.3b_${data_fraction}" \
    --file_path 'datasets/user_oriented_instructions.jsonl' \
    --save_file_name data_instruct/debug.jsonl; 

done