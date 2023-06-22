# train a small model for a few iterations
model_size=opt350m
model_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_sharded
model_config_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_hf
data_fraction=0.1
name=verify_epoch_iter_baseline
ckpt_path=/fs/cml-projects/instruction_following/$name
python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/dolly-train.json --added_tokens 0 \
--act_checkpointing --lr 2e-5 --max_steps 50 --accumulation_steps 4 --batch_size 16 \
--wandb --wb_project debug_epoch_iter --wrapped_class_name OPTDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name} \
--data_fraction $data_fraction --save_steps 10

# resume the saved model and continue training without fix
model_size=opt350m
model_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_sharded
model_config_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_hf
data_fraction=0.1
name=verify_epoch_iter_baseline_with_resume_no_fix
ckpt_path=/fs/cml-projects/instruction_following/$name
python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/dolly-train.json --added_tokens 0 \
--act_checkpointing --lr 2e-5 --max_steps 50 --accumulation_steps 4 --batch_size 16 \
--wandb --wb_project debug_epoch_iter --wrapped_class_name OPTDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name} \
--data_fraction $data_fraction --save_steps 15

python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/dolly-train.json --added_tokens 0 \
--act_checkpointing --lr 2e-5 --max_steps 50 --accumulation_steps 4 --batch_size 16 \
--wandb --wb_project debug_epoch_iter --wrapped_class_name OPTDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name} \
--data_fraction $data_fraction --save_steps 15 --resume


# resume the saved model and continue training without fix
model_size=opt350m
model_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_sharded
model_config_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_hf
data_fraction=0.1
name=verify_epoch_iter_baseline_with_resume_with_fix
ckpt_path=/fs/cml-projects/instruction_following/$name
python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/dolly-train.json --added_tokens 0 \
--act_checkpointing --lr 2e-5 --max_steps 50 --accumulation_steps 4 --batch_size 16 \
--wandb --wb_project debug_epoch_iter --wrapped_class_name OPTDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name} \
--data_fraction $data_fraction --save_steps 15

python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/dolly-train.json --added_tokens 0 \
--act_checkpointing --lr 2e-5 --max_steps 50 --accumulation_steps 4 --batch_size 16 \
--wandb --wb_project debug_epoch_iter --wrapped_class_name OPTDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name} \
--data_fraction $data_fraction --save_steps 15 --resume

# resume the saved model and continue training without fix
model_size=opt350m
model_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_sharded
model_config_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_hf
data_fraction=0.1
name=verify_epoch_iter_baseline_with_resume_with_alt_fix
ckpt_path=/fs/cml-projects/instruction_following/$name
python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/dolly-train.json --added_tokens 0 \
--act_checkpointing --lr 2e-5 --max_steps 50 --accumulation_steps 4 --batch_size 16 \
--wandb --wb_project debug_epoch_iter --wrapped_class_name OPTDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name} \
--data_fraction $data_fraction --save_steps 3

python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/dolly-train.json --added_tokens 0 \
--act_checkpointing --lr 2e-5 --max_steps 50 --accumulation_steps 4 --batch_size 16 \
--wandb --wb_project debug_epoch_iter --wrapped_class_name OPTDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name} \
--data_fraction $data_fraction --save_steps 100 --resume


