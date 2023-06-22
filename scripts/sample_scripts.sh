# script for converting the llama weights to hugging face format
python convert_llama_weights_to_hf.py --input_dir /fs/nexus-scratch/pchiang/llama --model_size 13B --output_dir /fs/nexus-scratch/pchiang/llama/13B_hf
# script for converting the hugging face format to fsdp format
python convert_hf_to_fsdp.py --load_path /fs/nexus-scratch/pchiang/llama/13B_hf --save_path /fs/nexus-scratch/pchiang/llama/13B_sharded

# script for converting opt model from the hugging face format to fsdp format
python convert_hf_to_fsdp.py --load_path facebook/opt-350m \
--save_path /fs/cml-projects/instruction_following/pretrained_models/opt350m_sharded \
--save_path_hf /fs/cml-projects/instruction_following/pretrained_models/opt350m_hf \
--cache_dir /fs/cml-projects/instruction_following/cache \
--add_tokens 0
python convert_hf_to_fsdp.py --load_path facebook/opt-1.3b \
--save_path /fs/cml-projects/instruction_following/pretrained_models/opt1.3b_sharded \
--save_path_hf /fs/cml-projects/instruction_following/pretrained_models/opt1.3b_hf \
--cache_dir /fs/cml-projects/instruction_following/cache \
--add_tokens 0
python convert_hf_to_fsdp.py --load_path facebook/opt-6.7b \
--save_path /fs/cml-projects/instruction_following/pretrained_models/opt6.7b_sharded \
--save_path_hf /fs/cml-projects/instruction_following/pretrained_models/opt6.7b_hf \
--cache_dir /fs/cml-projects/instruction_following/cache \
--add_tokens 0

# training script for opt350m on 2 rtxa5000 for 5 epochs 52000*5/128=2031
model_size=opt350m
model_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_sharded
model_config_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_hf
data_fraction=1.0
name=${model_size}_self-instruct_dataf${data_fraction}_fsdp
ckpt_path=/fs/cml-projects/instruction_following/$name
python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/alpaca.json --added_tokens 0 \
--act_checkpointing --lr 2e-5 --max_steps 2031 --accumulation_steps 4 --batch_size 16 \
--wandb --wb_project huggingface --wrapped_class_name OPTDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name}_v2 \
--data_fraction $data_fraction 

# training script for opt1.3b on 4 rtxa5000 for 5 epochs 52000*5/128=2031
model_size=opt1.3b
model_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_sharded
model_config_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_hf
data_fraction=1.0
name=${model_size}_self-instruct_dataf${data_fraction}_fsdp
ckpt_path=/fs/cml-projects/instruction_following/$name
python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/alpaca.json --added_tokens 0 \
--act_checkpointing --lr 2e-5 --max_steps 2031 --accumulation_steps 2 --batch_size 16 \
--wandb --wb_project huggingface --wrapped_class_name OPTDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name}_v2 \
--data_fraction $data_fraction --hack

# training script for opt6.7b on 4 rtxa5000 for 5 epochs 52000*5/128=2031
model_size=opt6.7b
model_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_sharded
model_config_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_hf
data_fraction=1.0
name=${model_size}_self-instruct_dataf${data_fraction}_fsdp
ckpt_path=/fs/cml-projects/instruction_following/$name
python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/alpaca.json --added_tokens 0 \
--act_checkpointing --lr 2e-5 --max_steps 2031 --accumulation_steps 4 --batch_size 8 \
--wandb --wb_project huggingface --wrapped_class_name OPTDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name}_v2 \
--data_fraction $data_fraction --hack


# training script for llama7B on 4 rtxa5000 for 5 epochs 52000*5/128=2031
model_size=7B
model_path=/fs/nexus-scratch/pchiang/llama/${model_size}_sharded
model_config_path=/fs/nexus-scratch/pchiang/llama/${model_size}_hf
data_fraction=1.0
name=llama${model_size}_self-instruct_dataf${data_fraction}_fsdp
ckpt_path=/fs/cml-projects/instruction_following/$name
python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/alpaca.json --added_tokens 1 \
--act_checkpointing --lr 2e-5 --max_steps 2031 --accumulation_steps 8 --batch_size 4 \
--wandb --wb_project huggingface --wrapped_class_name LlamaDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name}_v2 \
--data_fraction $data_fraction --hack


