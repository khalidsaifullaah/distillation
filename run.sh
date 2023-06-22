CKPT1_SHARDED_PATH=/cmlscratch/khalids/dalle_mini/redpajama_3B_sharded
CKPT1_HF_PATH=/cmlscratch/khalids/dalle_mini/redpajama_3B_hf

CKPT2_SHARDED_PATH=/cmlscratch/khalids/dalle_mini/redpajama_7B_chat_sharded
CKPT2_HF_PATH=/cmlscratch/khalids/dalle_mini/redpajama_7B_chat_hf
CACHE_PATH=/cmlscratch/khalids/dalle_mini/cache

DATA_PATH=datasets/alpaca-train.jsonl

# python convert_hf_to_fsdp.py \
# --load_path togethercomputer/RedPajama-INCITE-Base-3B-v1 \
# --save_path $CKPT_SHARDED_PATH \
# --save_path_hf $CKPT_HF_PATH \
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
CKPT_PATH=/cmlscratch/khalids/dalle_mini/redpajama_3B_logit_w_reverse_kl_loss_alpha_0.25_tmp_1
python train.py \
--init_checkpoint_path $CKPT1_SHARDED_PATH \
--model_config_path $CKPT1_HF_PATH \
--teacher_model_init_checkpoint_path $CKPT2_SHARDED_PATH \
--teacher_model_config_path $CKPT2_HF_PATH \
--checkpoint_path $CKPT_PATH \
--wrapped_class_name GPTNeoXLayer \
--data_path $DATA_PATH \
--hack --alpha 0.25 --tmp 1.0 --logit_distillation_mode --loss_type reverse_kl --wandb --wb_id bslifjaoeihgaaa --wb_name logit_w_reverse_kl_loss_alpha_0.25_tmp_1

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