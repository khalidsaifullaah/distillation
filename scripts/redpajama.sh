CKPT_SHARDED_PATH=/fs/cml-projects/instruction_following/pretrained_models/redpajama_sharded
CKPT_HF_PATH=/fs/cml-projects/instruction_following/pretrained_models/redpajama_hf
CKPT_PATH=/fs/cml-projects/instruction_following/redpajama
CACHE_PATH=/fs/cml-projects/instruction_following/pretrained_models/cache
DATA_PATH=datasets/alpaca-train.jsonl

python convert_hf_to_fsdp.py \
--load_path togethercomputer/RedPajama-INCITE-Base-3B-v1 \
--save_path $CKPT_SHARDED_PATH \
--save_path_hf $CKPT_HF_PATH \
--cache_dir $CACHE_PATH \
--add_tokens 0

python train.py \
--init_checkpoint_path $CKPT_SHARDED_PATH \
--model_config_path $CKPT_HF_PATH \
--checkpoint_path $CKPT_PATH \
--wrapped_class_name GPTNeoXLayer \
--data_path $DATA_PATH \
--data_fraction 0.1 --hack