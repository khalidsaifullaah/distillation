# python eval_generate.py --dummy_path /fs/nexus-scratch/pchiang/llama/7B_hf \
# --model /fs/cml-projects/instruction_following/llama7B_self-instruct_dataf0.2_fsdp/2030/model \
# --template_type alpaca \
# --file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_llama7B-self-instruct-df0.2.jsonl --debug

# python eval_generate.py --dummy_path /fs/nexus-scratch/pchiang/llama/7B_hf \
# --model /fs/cml-projects/instruction_following/llama7B_self-instruct_dataf0.4_fsdp/2030/model \
# --template_type alpaca \
# --file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_llama7B-self-instruct-df0.4.jsonl --debug

# python eval_generate.py --dummy_path /fs/nexus-scratch/pchiang/llama/7B_hf \
# --model /fs/cml-projects/instruction_following/llama7B_self-instruct_dataf0.6_fsdp/2030/model \
# --template_type alpaca \
# --file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_llama7B-self-instruct-df0.6.jsonl --debug

# python eval_generate.py --dummy_path /fs/nexus-scratch/pchiang/llama/7B_hf \
# --model /fs/cml-projects/instruction_following/llama7B_self-instruct_dataf0.8_fsdp/2030/model \
# --template_type alpaca \
# --file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_llama7B-self-instruct-df0.8.jsonl --debug

# python eval_generate.py --dummy_path /fs/nexus-scratch/pchiang/llama/7B_hf \
# --model /fs/cml-projects/instruction_following/llama7B_self-instruct_dataf1.0_fsdp/2030/model \
# --template_type alpaca \
# --file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_llama7B-self-instruct-df1.0.jsonl --debug



python eval_generate.py --dummy_path /fs/nexus-scratch/pchiang/llama/7B_hf \
--model /fs/cml-projects/instruction_following/alpaca-llama7B_self-instruct_dataf0.2/1935/model \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_llama7B-self-instruct-df0.2_prior.jsonl --debug

python eval_generate.py --dummy_path /fs/nexus-scratch/pchiang/llama/7B_hf \
--model /fs/cml-projects/instruction_following/alpaca-llama7B_self-instruct_dataf0.4/1935/model \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_llama7B-self-instruct-df0.4_prior.jsonl --debug

python eval_generate.py --dummy_path /fs/nexus-scratch/pchiang/llama/7B_hf \
--model /fs/cml-projects/instruction_following/alpaca-llama7B_self-instruct_dataf0.6/1935/model \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_llama7B-self-instruct-df0.6_prior.jsonl --debug

python eval_generate.py --dummy_path /fs/nexus-scratch/pchiang/llama/7B_hf \
--model /fs/cml-projects/instruction_following/alpaca-llama7B_self-instruct_dataf0.8/1935/model \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_llama7B-self-instruct-df0.8_prior.jsonl --debug

python eval_generate.py --dummy_path /fs/nexus-scratch/pchiang/llama/7B_hf \
--model /fs/cml-projects/instruction_following/alpaca-llama7B_self-instruct_dataf1.0/1935/model \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_llama7B-self-instruct-df1.0_prior.jsonl --debug