# latest runs with quadruple learning rate

python eval_generate.py --dummy_path "/fs/cml-projects/instruction_following/pretrained_models/opt6.7b_hf" \
--model "/fs/cml-projects/instruction_following/opt6.7b_self-instruct_dataf0.2_fsdp/2030/model" \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_opt6.7B-self-instruct-df0.2.jsonl --debug

python eval_generate.py --dummy_path "/fs/cml-projects/instruction_following/pretrained_models/opt6.7b_hf" \
--model "/fs/cml-projects/instruction_following/opt6.7b_self-instruct_dataf0.4_fsdp/2030/model" \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_opt6.7B-self-instruct-df0.4.jsonl --debug

python eval_generate.py --dummy_path "/fs/cml-projects/instruction_following/pretrained_models/opt6.7b_hf" \
--model "/fs/cml-projects/instruction_following/opt6.7b_self-instruct_dataf0.6_fsdp/2030/model" \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_opt6.7B-self-instruct-df0.6.jsonl --debug

python eval_generate.py --dummy_path "/fs/cml-projects/instruction_following/pretrained_models/opt6.7b_hf" \
--model "/fs/cml-projects/instruction_following/opt6.7b_self-instruct_dataf0.8_fsdp/2030/model" \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_opt6.7B-self-instruct-df0.8.jsonl --debug

python eval_generate.py --dummy_path "/fs/cml-projects/instruction_following/pretrained_models/opt6.7b_hf" \
--model "/fs/cml-projects/instruction_following/opt6.7b_self-instruct_dataf1.0_fsdp/2030/model" \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_opt6.7B-self-instruct-df1.0.jsonl --debug

# prior runs
python eval_generate.py --dummy_path "/fs/cml-projects/instruction_following/pretrained_models/opt6.7b_hf" \
--model "/fs/cml-projects/instruction_following/alpaca-opt-6.7b_dataf0.8/2030/model" \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_opt6.7B-self-instruct-df0.8_prior.jsonl --debug

python eval_generate.py --dummy_path "/fs/cml-projects/instruction_following/pretrained_models/opt6.7b_hf" \
--model "/fs/cml-projects/instruction_following/alpaca-opt-6.7b_dataf1.0/2030/model" \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_opt6.7B-self-instruct-df1.0_prior.jsonl --debug

# plain opt
python eval_generate.py --dummy_path "/fs/cml-projects/instruction_following/pretrained_models/opt6.7b_hf" \
--model "/fs/cml-projects/instruction_following/pretrained_models/opt6.7b_sharded" \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_opt6.7B.jsonl --debug


