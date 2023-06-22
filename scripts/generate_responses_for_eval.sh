python eval_generate.py --dummy_path "/fs/nexus-scratch/pchiang/llama/7B_hf" \
--model "/fs/nexus-scratch/pchiang/llama/7B_sharded" \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/vicuna_llama7B.jsonl

python eval_generate.py --dummy_path "/fs/nexus-scratch/pchiang/llama/7B_hf" \
--model "/fs/nexus-scratch/pchiang/llama/7B_sharded" \
--template_type alpaca \
--file_path datasets/dolly-val\(processed\).jsonl --save_file_name outputs/answers/dolly_llama7B.jsonl

python eval_generate.py --dummy_path "/fs/nexus-scratch/pchiang/llama/7B_hf" \
--model "/fs/nexus-scratch/pchiang/llama/7B_sharded" \
--template_type alpaca \
--file_path datasets/self-instruct-val\(processed\).jsonl --save_file_name outputs/answers/self-instruct_llama7B.jsonl

python eval_generate.py --dummy_path "/fs/cml-projects/instruction_following/pretrained_models/opt6.7b_hf" \
--model "/fs/cml-projects/instruction_following/opt6.7b_self-instruct_dataf1.0_fsdp/2030/model" \
--template_type alpaca \
--file_path datasets/vicuna-val.jsonl --save_file_name outputs/answers/self-instruct_opt6.7B-self-instruct-df1.0.jsonl

