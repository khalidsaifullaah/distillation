# CKPT1_SHARDED_PATH=/cmlscratch/khalids/dalle_mini/redpajama_3B_sharded
# CKPT1_HF_PATH=/cmlscratch/khalids/dalle_mini/redpajama_3B_hf

# CKPT2_SHARDED_PATH=/cmlscratch/khalids/dalle_mini/redpajama_7B_chat_sharded
# CKPT2_HF_PATH=/cmlscratch/khalids/dalle_mini/redpajama_7B_chat_hf
# CACHE_PATH=/cmlscratch/khalids/dalle_mini/cache

# DATA_PATH=datasets/alpaca-train.jsonl

# python convert_hf_to_fsdp.py \
# --load_path togethercomputer/RedPajama-INCITE-Base-3B-v1 \
# --save_path $CKPT1_SHARDED_PATH \
# --save_path_hf $CKPT1_HF_PATH \
# --cache_dir $CACHE_PATH \
# --add_tokens 0

# python convert_hf_to_fsdp.py \
# --load_path togethercomputer/RedPajama-INCITE-7B-Chat \
# --save_path $CKPT2_SHARDED_PATH \
# --save_path_hf $CKPT2_HF_PATH \
# --cache_dir $CACHE_PATH \
# --add_tokens 0

# logit distillation with kl loss
# CKPT_PATH=/cmlscratch/khalids/dalle_mini/redpajama_3B_logit_w_kl_loss_alpha_0.25
# python train.py \
# --init_checkpoint_path $CKPT1_SHARDED_PATH \
# --model_config_path $CKPT1_HF_PATH \
# --teacher_model_init_checkpoint_path $CKPT2_SHARDED_PATH \
# --teacher_model_config_path $CKPT2_HF_PATH \
# --checkpoint_path $CKPT_PATH \
# --wrapped_class_name GPTNeoXLayer \
# --data_path $DATA_PATH \
# --hack --alpha 0.25 --logit_distillation_mode --wandb --wb_id bdslifjaoeihgaaa --wb_name logit_w_kl_loss_alpha_0.25

# logit distillation with reverse_kl loss
# CKPT_PATH=/cmlscratch/khalids/dalle_mini/redpajama_3B_logit_w_reverse_kl_loss_alpha_0.25
# python train.py \
# --init_checkpoint_path $CKPT1_SHARDED_PATH \
# --model_config_path $CKPT1_HF_PATH \
# --teacher_model_init_checkpoint_path $CKPT2_SHARDED_PATH \
# --teacher_model_config_path $CKPT2_HF_PATH \
# --checkpoint_path $CKPT_PATH \
# --wrapped_class_name GPTNeoXLayer \
# --data_path $DATA_PATH \
# --hack --alpha 0.25 --logit_distillation_mode --loss_type reverse_kl --wandb --wb_id bjslifjaoeihgaaa --wb_name logit_w_reverse_kl_loss_alpha_0.25

# logit distillation with reverse_kl loss
# CKPT_PATH=/cmlscratch/khalids/dalle_mini/redpajama_3B_logit_w_reverse_kl_loss_alpha_0.25_tmp_1
# python train.py \
# --init_checkpoint_path $CKPT1_SHARDED_PATH \
# --model_config_path $CKPT1_HF_PATH \
# --teacher_model_init_checkpoint_path $CKPT2_SHARDED_PATH \
# --teacher_model_config_path $CKPT2_HF_PATH \
# --checkpoint_path $CKPT_PATH \
# --wrapped_class_name GPTNeoXLayer \
# --data_path $DATA_PATH \
# --hack --alpha 0.25 --tmp 1.0 --logit_distillation_mode --loss_type reverse_kl --wandb --wb_id bslifjaoeihgaaa --wb_name logit_w_reverse_kl_loss_alpha_0.25_tmp_1

# logit distillation with ce loss
# CKPT_PATH=/cmlscratch/khalids/dalle_mini/redpajama_3B_logit_w_ce_loss_alpha_0.25
# python train.py \
# --init_checkpoint_path $CKPT1_SHARDED_PATH \
# --model_config_path $CKPT1_HF_PATH \
# --teacher_model_init_checkpoint_path $CKPT2_SHARDED_PATH \
# --teacher_model_config_path $CKPT2_HF_PATH \
# --checkpoint_path $CKPT_PATH \
# --wrapped_class_name GPTNeoXLayer \
# --data_path $DATA_PATH \
# --hack --alpha 0.25 --logit_distillation_mode --loss_type ce --wandb --wb_id bfslifjaoeihgaaa --wb_name logit_w_ce_loss_alpha_0.25

# # baseline training
# CKPT_PATH=/cmlscratch/khalids/dalle_mini/redpajama_3B_baseline
# python train.py \
# --init_checkpoint_path $CKPT1_SHARDED_PATH \
# --model_config_path $CKPT1_HF_PATH \
# --checkpoint_path $CKPT_PATH \
# --wrapped_class_name GPTNeoXLayer \
# --data_path $DATA_PATH \
# --hack --wandb --wb_id beslifjaoeihgaaa --wb_name baseline


### MPT
# CKPT1_SHARDED_PATH=/cmlscratch/khalids/dalle_mini/mpt_1B_sharded
# CKPT1_HF_PATH=/cmlscratch/khalids/dalle_mini/mpt_1B_hf

# CKPT2_SHARDED_PATH=/cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded
# CKPT2_HF_PATH=/cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf
# CACHE_PATH=/cmlscratch/khalids/dalle_mini/cache

# DATA_PATH=datasets/alpaca-train.jsonl

# python convert_hf_to_fsdp.py \
# --load_path mosaicml/mpt-1b-redpajama-200b \
# --save_path $CKPT1_SHARDED_PATH \
# --save_path_hf $CKPT1_HF_PATH \
# --cache_dir $CACHE_PATH \
# --add_tokens 0

# python convert_hf_to_fsdp.py \
# --load_path mosaicml/mpt-7b-chat \
# --save_path $CKPT2_SHARDED_PATH \
# --save_path_hf $CKPT2_HF_PATH \
# --cache_dir $CACHE_PATH \
# --add_tokens 0

# MPT 1B baseline trained with MPT 7B data
# CKPT_PATH=/cmlscratch/khalids/dalle_mini/mpt_1B_teacher_mpt_7B_baseline
# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_teacher_mpt_7B_baseline \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack  --wandb --wb_id bitmifjaoeihgaaa --wb_name mpt_1B_teacher_mpt_7B_baseline

# logit distillation with reverse_kl loss
# CKPT_PATH=/cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_kl_loss_alpha_0.25_tmp_1
# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_kl_loss_alpha_0.25 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --logit_distillation_mode --loss_type reverse_kl --wandb --wb_name mpt_1B_logit_w_reverse_kl_loss_alpha_0.25

# logit distillation with kl loss
# CKPT_PATH=/cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_kl_loss_alpha_0.25_tmp_0.7
# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_kl_loss_alpha_0.25_tmp_0.7 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --logit_distillation_mode --wandb --wb_name mpt_1B_logit_w_kl_loss_alpha_0.25_tmp_0.7

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_masked_kl_sum_reduc_loss_alpha_0.25 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 1.0 --logit_distillation_mode --wandb --wb_name mpt_1B_logit_w_masked_kl_sum_reduc_loss_alpha_0.25

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_masked_forward_kl_mean_reduc_loss_alpha_0.25 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 1.0 --logit_distillation_mode --loss_type kl --wandb --wb_name mpt_1B_logit_w_masked_forward_kl_mean_reduc_loss_alpha_0.25


# logit distillation with bergman_div loss
# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_bergman_div_loss_alpha_0.25_z_0.1 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 0.1 --logit_distillation_mode --loss_type bergman_div --wandb --wb_name mpt_1B_logit_w_bergman_div_loss_alpha_0.25_z_0.1

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_bergman_div_loss_alpha_0.25_z_0.2 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 0.2 --logit_distillation_mode --loss_type bergman_div --wandb --wb_name mpt_1B_logit_w_bergman_div_loss_alpha_0.25_z_0.2

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_bergman_div_loss_alpha_0.25_z_0.3 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 0.3 --logit_distillation_mode --loss_type bergman_div --wandb --wb_name mpt_1B_logit_w_bergman_div_loss_alpha_0.25_z_0.3

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_bergman_div_loss_alpha_0.25_z_0.5 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 0.5 --logit_distillation_mode --loss_type bergman_div --wandb --wb_name mpt_1B_logit_w_bergman_div_loss_alpha_0.25_z_0.5

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_bergman_div_loss_alpha_0.25_z_0.7 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 0.7 --logit_distillation_mode --loss_type bergman_div --wandb --wb_name mpt_1B_logit_w_bergman_div_loss_alpha_0.25_z_0.7

# logit distillation with reverse_bergman_div loss
# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.25_z_0.1 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 0.1 --logit_distillation_mode --loss_type reverse_bergman_div --wandb --wb_name mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.25_z_0.1

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.25_z_0.2 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 0.2 --logit_distillation_mode --loss_type reverse_bergman_div --wandb --wb_name mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.25_z_0.2

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.25_z_0.3 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 0.3 --logit_distillation_mode --loss_type reverse_bergman_div --wandb --wb_name mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.25_z_0.3

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.25_z_0.5 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 0.5 --logit_distillation_mode --loss_type reverse_bergman_div --wandb --wb_name mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.25_z_0.5

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.25_z_0.7 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 0.7 --logit_distillation_mode --loss_type reverse_bergman_div --wandb --wb_name mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.25_z_0.7

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.2_z_0.1 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.2 --tmp 0.1 --logit_distillation_mode --loss_type reverse_bergman_div --wandb --wb_name mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.2_z_0.1

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.3_z_0.1 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.3 --tmp 0.1 --logit_distillation_mode --loss_type reverse_bergman_div --wandb --wb_name mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.3_z_0.1

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.5_z_0.1 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.5 --tmp 0.1 --logit_distillation_mode --loss_type reverse_bergman_div --wandb --wb_name mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.5_z_0.1

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.7_z_0.1 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.7 --tmp 0.1 --logit_distillation_mode --loss_type reverse_bergman_div --wandb --wb_name mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.7_z_0.1

# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.15_z_0.1 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.15 --tmp 0.1 --logit_distillation_mode --loss_type reverse_bergman_div --wandb --wb_name mpt_1B_logit_w_reverse_bergman_div_loss_alpha_0.15_z_0.1

# logit distillation with miniLLM loss
# python train.py \
# --init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_sharded \
# --model_config_path /cmlscratch/khalids/dalle_mini/mpt_1B_hf \
# --teacher_model_init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_sharded \
# --teacher_model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_chat_hf \
# --checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_1B_logit_w_miniLLM_reverse_kl_loss_alpha_0.25 \
# --wrapped_class_name GPTBlock \
# --data_path datasets/mpt_chat_data.jsonl \
# --hack --alpha 0.25 --tmp 1.0 --logit_distillation_mode --loss_type reverse_kl --wandb --wb_name mpt_1B_logit_w_miniLLM_reverse_kl_loss_alpha_0.25


# dolphin data baseline training w/ MPT 7B
python train.py \
--init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_sharded \
--model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_hf \
--checkpoint_path /cmlscratch/khalids/dalle_mini/dolphin_mpt_7B_dfrac_1_baseline \
--wrapped_class_name GPTBlock \
--data_path datasets/dolphin.jsonl \
--hack  --data_fraction 0.01 --batch_size 1 --accumulation_steps 8 --wandb --wb_name dolphin_mpt_7B_dfrac_1_baseline

python train.py \
--init_checkpoint_path /cmlscratch/khalids/dalle_mini/mpt_7B_sharded \
--model_config_path /cmlscratch/khalids/dalle_mini/mpt_7B_hf \
--checkpoint_path /cmlscratch/khalids/dalle_mini/dolphin_mpt_7B_dfrac_10_baseline \
--wrapped_class_name GPTBlock \
--data_path datasets/dolphin.jsonl \
--hack  --data_fraction 0.10 --batch_size 1 --accumulation_steps 8 --wandb --wb_name dolphin_mpt_7B_dfrac_10_baseline

# Machine data (Alpaca) baseline training
# CKPT_PATH=/cmlscratch/khalids/dalle_mini/mpt_1B_alpaca_baseline
# python train.py \
# --init_checkpoint_path $CKPT1_SHARDED_PATH \
# --model_config_path $CKPT1_HF_PATH \
# --checkpoint_path $CKPT_PATH \
# --wrapped_class_name GPTBlock \
# --data_path $DATA_PATH \
# --hack  --wandb --wb_id bhtmifjaoeihgaaa --wb_name baseline

# Human data (Dolly) baseline training
# CKPT_PATH=/cmlscratch/khalids/dalle_mini/mpt_1B_dolly_baseline
# DATA_PATH=datasets/dolly-train.jsonl
# python train.py \
# --init_checkpoint_path $CKPT1_SHARDED_PATH \
# --model_config_path $CKPT1_HF_PATH \
# --checkpoint_path $CKPT_PATH \
# --wrapped_class_name GPTBlock \
# --data_path $DATA_PATH \
# --hack 
# --wandb --wb_name mpt_1B_dolly_baseline