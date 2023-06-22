# this runs training on the 350m model with and without our lr hack, which artificially increase lr by 4 times
# this should show that the hack does not influence the training curve

python train.py --init_checkpoint_path /fs/cml-projects/instruction_following/pretrained_models/opt350m_sharded \
--model_config_path /fs/cml-projects/instruction_following/pretrained_models/opt350m_hf \
--act_checkpointing --lr 2e-5 --max_steps 1500 --accumulation_steps 2 --batch_size 16 \
--wandb --wb_project debug_bf16 --wrapped_class_name OPTDecoderLayer \
--checkpoint_path /fs/cml-projects/instruction_following/test \
--wb_name opt-350m_dolly_1000steps_lr2e-5 --wb_id debug_aaaaea \
--data_fraction 1.0 --data_path data_instruct/dolly-train.json --added_tokens 0 --max_grad_norm 10000

python train.py --init_checkpoint_path /fs/cml-projects/instruction_following/pretrained_models/opt350m_sharded \
--model_config_path /fs/cml-projects/instruction_following/pretrained_models/opt350m_hf \
--act_checkpointing --lr 2e-5 --max_steps 1500 --accumulation_steps 2 --batch_size 16 \
--wandb --wb_project debug_bf16 --wrapped_class_name OPTDecoderLayer \
--checkpoint_path /fs/cml-projects/instruction_following/test \
--wb_name opt-350m_dolly_1000steps_lr2e-5_hack --wb_id debug_aaaaeb \
--data_fraction 1.0 --data_path data_instruct/dolly-train.json --added_tokens 0 --max_grad_norm 10000 --hack

python train.py --init_checkpoint_path /fs/cml-projects/instruction_following/pretrained_models/opt350m_sharded \
--model_config_path /fs/cml-projects/instruction_following/pretrained_models/opt350m_hf \
--act_checkpointing --lr 1e-5 --max_steps 1500 --accumulation_steps 2 --batch_size 16 \
--wandb --wb_project debug_bf16 --wrapped_class_name OPTDecoderLayer \
--checkpoint_path /fs/cml-projects/instruction_following/test \
--wb_name opt-350m_dolly_1000steps_lr1e-5_hack --wb_id debug_aaaaec \
--data_fraction 1.0 --data_path data_instruct/dolly-train.json --added_tokens 0 --max_grad_norm 10000 --hack

python train.py --init_checkpoint_path /fs/cml-projects/instruction_following/pretrained_models/opt350m_sharded \
--model_config_path /fs/cml-projects/instruction_following/pretrained_models/opt350m_hf \
--act_checkpointing --lr 5e-6 --max_steps 1500 --accumulation_steps 2 --batch_size 16 \
--wandb --wb_project debug_bf16 --wrapped_class_name OPTDecoderLayer \
--checkpoint_path /fs/cml-projects/instruction_following/test \
--wb_name opt-350m_dolly_1000steps_lr2e-5_hack --wb_id debug_aaaaed \
--data_fraction 1.0 --data_path data_instruct/dolly-train.json --added_tokens 0 --max_grad_norm 10000 --hack