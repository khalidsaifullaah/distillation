python model_eval_vicuna.py -q vicuna_prompts/question.jsonl \
-a outputs/answers/vicuna_llama7B.jsonl outputs/answers/vicuna_opt6.7B.jsonl \
-p vicuna_prompts/prompt.jsonl -r vicuna_prompts/reviewer.jsonl -o outputs/reviews/llama7B_vs_opt6.7B.jsonl