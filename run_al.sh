# TODO: make a mixed_sampling run with 50-50
# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.001 --cluster_data_fraction 0.01 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_forward_ppl

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.002 --cluster_data_fraction 0.01 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_forward_ppl

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.003 --cluster_data_fraction 0.01 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.003_random_poolfrac_0.01_forward_ppl

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.004 --cluster_data_fraction 0.01 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.004_random_poolfrac_0.01_forward_ppl

# python eval_generate.py --model_config_path /sensei-fs/users/ksaifullah/al_dolphin_llama2_7B_dfrac_0.004_random_poolfrac_0.01_forward_ppl_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.004_random_poolfrac_0.01_forward_ppl.json

# python eval_generate.py --model_config_path /sensei-fs/users/ksaifullah/dolphin_llama2_7B_dfrac_5_random_hf/ --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_5_random.json

# python eval_generate.py --model_config_path /sensei-fs/users/ksaifullah/dolphin_llama2_7B_dfrac_10_random_hf/ --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_10_random.json

# python eval_generate.py --model_config_path /sensei-fs/users/ksaifullah/dolphin_llama2_7B_dfrac_5_cluster_hf/ --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_5_cluster.json

# export ANTHROPIC_API_KEY=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

# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.004_random_poolfrac_0.01_forward_ppl.json --annotators_config 'claude'

# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_5_random.json --annotators_config 'claude'

# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_10_random.json --annotators_config 'claude'

# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_5_cluster.json --annotators_config 'claude'
# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.001 --cluster_data_fraction 0.01 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_forward_ppl

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.005 --cluster_data_fraction 0.01 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.005_random_poolfrac_0.01_forward_ppl  --resume --resume_checkpoint_path ../al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_forward_ppl_sharded

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.001 --cluster_data_fraction 0.01 --random_pool_fraction --mixed_sampling --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_mixed_sample_forward_ppl
# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.001 --cluster_data_fraction 0.01 --random_pool_fraction --mixed_sampling --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_mixed_sampling_0.7_forward_ppl

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.001 --cluster_data_fraction 0.01 --random_pool_fraction --model_ask --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_model_ask_forward_ppl

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.002 --cluster_data_fraction 0.01 --random_pool_fraction --mixed_sampling --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_mixed_sampling_0.5_forward_ppl

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.002 --cluster_data_fraction 0.01

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.005 --cluster_data_fraction 0.01

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.01 --cluster_data_fraction 0.01

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.02 --cluster_data_fraction 0.01

# python main.py --model hf-causal --model_args pretrained=/sensei-fs/users/ksaifullah/dolphin_llama2_7B_dfrac_5_random_hf/,dtype='float16' --tasks hellaswag,winogrande,boolq,piqa,arc_challenge --device cuda:0 --no_cache

# python main.py --model hf-causal --model_args pretrained=/sensei-fs/users/ksaifullah/dolphin_llama2_7B_dfrac_10_random_hf/,dtype='float16' --tasks hellaswag,winogrande,boolq,piqa,arc_challenge --device cuda:0 --no_cache

# python main.py --model hf-causal --model_args pretrained=/sensei-fs/users/ksaifullah/dolphin_llama2_7B_dfrac_50_random_hf/,dtype='float16' --tasks hellaswag,winogrande,boolq,piqa,arc_challenge --device cuda:1 --no_cache

# python main.py --model hf-causal --model_args pretrained=/sensei-fs/users/ksaifullah/dolphin_llama2_7B_dfrac_100_random_hf/,dtype='float16' --tasks hellaswag,winogrande,boolq,piqa,arc_challenge --device cuda:1 --no_cache

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.001 --cluster_data_fraction 0.01 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_forward_ppl

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_forward_ppl

# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_forward_ppl.json

# export ANTHROPIC_API_KEY=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

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_bottom_forward_ppl

# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_bottom_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.9_bottom_forward_ppl

# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.9_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.9_bottom_forward_ppl.json


# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.001 --num_acquisition_samples 100 --random_pool_fraction --sampling_strategy generation_ppl --mixed_sampling --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.001_mixed_sampling_generation_ppl

# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.001_mixed_sampling_generation_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.001_mixed_sampling_generation_ppl.json


# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.001 --num_acquisition_samples 100 --random_pool_fraction --sampling_strategy generation_ppl --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.001_generation_ppl

# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.001_generation_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.001_generation_ppl.json


# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_forward_ppl.json --annotators_config claude

# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_bottom_forward_ppl.json --annotators_config claude

# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.001_generation_ppl.json --annotators_config claude

# python eval_generate.py --sharded_model ../dolphin_llama2_7B_dfrac_0.01_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_0.01_random.json

# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_0.01_random.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.01 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.1 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.3 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.3_bottom_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.3_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.3_bottom_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.5 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.5_bottom_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.5_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.5_bottom_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.7 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.7_bottom_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.7_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.7_bottom_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.9 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.9_bottom_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.9_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.9_bottom_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.99 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.99_bottom_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.99_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.99_bottom_forward_ppl.json

# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.3_bottom_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.5_bottom_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.7_bottom_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.9_bottom_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.99_bottom_forward_ppl.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --mixed_sampling --mixed_sampling_factor 0.1 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_cluster_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_cluster_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_cluster_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl.json
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_cluster_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --sampling_strategy cluster --num_acquisition_samples 100 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_cluster_sampling
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_cluster_sampling_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_cluster_sampling.json
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_cluster_sampling.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --sampling_strategy random --num_acquisition_samples 100 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_sampling --resume --resume_checkpoint_path ../al_dolphin_llama2_7B_dfrac_0.0001_random_sampling_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_sampling_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_sampling.json
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_sampling.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --sampling_strategy random --num_acquisition_samples 373 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_sampling_once
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_sampling_once_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_sampling_once.json
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_sampling_once.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --sampling_strategy random --num_acquisition_samples 100 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_sampling
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_sampling_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_sampling.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.1 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_forward_ppl.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ~/dolphin_llama2_7B_dfrac_0.02_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method random --dont_save_opt --num_epochs 2 --data_fraction 0.0002 --batch_size 1 --accumulation_steps 8
# python eval_generate.py --sharded_model ../dolphin_llama2_7B_dfrac_0.02_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_0.02_random.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_instruct_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_instruct_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_instruct_forward_ppl.json

# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_random_sampling.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_instruct_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_0.02_random.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_fixed_random_poolfrac_0.01_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_fixed_random_poolfrac_0.01_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_fixed_random_poolfrac_0.01_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_fixed_random_poolfrac_0.01_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_fixed_random_poolfrac_0.01_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_fixed_random_poolfrac_0.01_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0005 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_forward_ppl.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ~/dolphin_llama2_7B_dfrac_0.05_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method random --dont_save_opt --num_epochs 2 --data_fraction 0.0005 --batch_size 1 --accumulation_steps 8
# python eval_generate.py --sharded_model ../dolphin_llama2_7B_dfrac_0.05_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_0.05_random.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_instruct_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_instruct_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_instruct_forward_ppl.json
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_instruct_forward_ppl.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0005 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_instruct_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_instruct_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_instruct_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0008 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_instruct_forward_ppl --resume --resume_checkpoint_path ../al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_instruct_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_instruct_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_instruct_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_instruct_forward_ppl --resume --resume_checkpoint_path ../al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_instruct_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_instruct_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_instruct_forward_ppl.json

# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_instruct_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_instruct_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_instruct_forward_ppl.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_instruct_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_instruct_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_instruct_forward_ppl.json
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_instruct_forward_ppl.json --annotators_config claude

# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_fixed_random_poolfrac_0.01_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_fixed_random_poolfrac_0.01_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_0.05_random.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 3 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_3_w_decay_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_3_w_decay_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_3_w_decay_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 3 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_3_w_decay_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_3_w_decay_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_3_w_decay_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 3 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_3_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_3_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_3_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0005 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0005 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 3 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_3_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_3_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_3_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0005 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

export ANTHROPIC_API_KEY=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

# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_3_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_7_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_2_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_3_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_3_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0008 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0005 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0008 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json --resume --resume_checkpoint_path ../al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --stratification_strategy bucket --num_k 5 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_cluster_poolfrac_0.01_bucket_stratify_5_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.0001_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --seed 84 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --stratification_strategy bucket --num_k 2 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0002_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.0002_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --resume --resume_checkpoint_path ../al_dolphin_llama2_7B_dfrac_0.0001_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0002_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0002_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --seed 168 --al_data_fraction 0.0005 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --stratification_strategy bucket --num_k 2 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0005_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.0005_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --resume --resume_checkpoint_path ../al_dolphin_llama2_7B_dfrac_0.0002_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0005_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0005_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --seed 336 --al_data_fraction 0.0008 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --stratification_strategy bucket --num_k 2 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0008_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.0008_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --resume --resume_checkpoint_path ../al_dolphin_llama2_7B_dfrac_0.0005_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0008_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0008_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_3_w_decay_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_3_w_decay_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_forward_ppl.json --annotators_config claude

# python train.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ../dolphin_llama2_7B_dfrac_0.01_cluster \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method cluster  --data_fraction 0.0001 --batch_size 1 --accumulation_steps 8 
# # --wandb --wb_name dolphin_llama2_7B_dfrac_0.01_cluster

# python train.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ../dolphin_llama2_7B_dfrac_0.02_cluster \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method cluster  --data_fraction 0.0002 --batch_size 1 --accumulation_steps 8 
# # --wandb --wb_name dolphin_llama2_7B_dfrac_0.02_cluster

# python train.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ../dolphin_llama2_7B_dfrac_0.05_cluster \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method cluster  --data_fraction 0.0005 --batch_size 1 --accumulation_steps 8 
# # --wandb --wb_name dolphin_llama2_7B_dfrac_0.05_cluster

# python train.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ../dolphin_llama2_7B_dfrac_0.08_cluster \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method cluster  --data_fraction 0.0008 --batch_size 1 --accumulation_steps 8 
# # --wandb --wb_name dolphin_llama2_7B_dfrac_0.08_cluster

# python train.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ../dolphin_llama2_7B_dfrac_0.1_cluster \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method cluster  --data_fraction 0.001 --batch_size 1 --accumulation_steps 8 
# # --wandb --wb_name dolphin_llama2_7B_dfrac_0.1_cluster

# python eval_generate.py --sharded_model ../dolphin_llama2_7B_dfrac_0.01_cluster --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_0.01_cluster.json
# python eval_generate.py --sharded_model ../dolphin_llama2_7B_dfrac_0.02_cluster --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_0.02_cluster.json
# python eval_generate.py --sharded_model ../dolphin_llama2_7B_dfrac_0.05_cluster --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_0.05_cluster.json
# python eval_generate.py --sharded_model ../dolphin_llama2_7B_dfrac_0.08_cluster --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_0.08_cluster.json
# python eval_generate.py --sharded_model ../dolphin_llama2_7B_dfrac_0.1_cluster --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_0.1_cluster.json

# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_0.01_cluster.json --annotators_config claude
# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_0.02_cluster.json --annotators_config claude
# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_0.05_cluster.json --annotators_config claude
# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_0.08_cluster.json --annotators_config claude
# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_0.1_cluster.json --annotators_config claude


# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0002_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0005_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0008_cluster_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config claude

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.001 --cluster_data_fraction 0.01 --num_acquisition_samples 1000 --random_pool_fraction --stratification_strategy bucket --num_k 3 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.002 --cluster_data_fraction 0.01 --num_acquisition_samples 1000 --random_pool_fraction --stratification_strategy bucket --num_k 3 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.005 --cluster_data_fraction 0.01 --num_acquisition_samples 1000 --random_pool_fraction --stratification_strategy bucket --num_k 1 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.005_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.005_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json --resume --resume_checkpoint_path ../al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.005_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.005_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.008 --cluster_data_fraction 0.01 --num_acquisition_samples 1000 --random_pool_fraction --stratification_strategy bucket --num_k 1 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.008_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.008_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json --resume --resume_checkpoint_path ../al_dolphin_llama2_7B_dfrac_0.005_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.008_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.008_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.002 --cluster_data_fraction 0.01 --num_acquisition_samples 1000 --random_pool_fraction --stratification_strategy bucket --num_k 5 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.005 --cluster_data_fraction 0.01 --num_acquisition_samples 1000 --random_pool_fraction --stratification_strategy bucket --num_k 5 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.005_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.005_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.005_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.005_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --batch_size 4 --al_data_fraction 0.008 --cluster_data_fraction 0.01 --num_acquisition_samples 1000 --random_pool_fraction --stratification_strategy bucket --num_k 5 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.008_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_7B_dfrac_0.008_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.008_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.008_random_poolfrac_0.01_acquisition_1000_bucket_stratify_5_w_decay2_forward_ppl.json

# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.001_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.002_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.005_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json --annotators_config claude
# alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.008_random_poolfrac_0.01_acquisition_1000_bucket_stratify_3_w_decay_forward_ppl.json --annotators_config claude

# python eval_generate.py --hf_model_path /sensei-fs/users/ksaifullah/dolphin_llama2_7B_dfrac_100_random_hf --file_path alpaca_eval --save_file_name dolphin_llama2_7B_dfrac_100_random.json
# alpaca_eval --model_outputs dolphin_llama2_7B_dfrac_100_random.json --annotators_config claude

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_13B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf \
# --checkpoint_path ../dolphin_llama2_13B_dfrac_0.01_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method random --dont_save_opt --num_epochs 2 --lr 5e-5 --data_fraction 0.0001 --batch_size 1 --accumulation_steps 8 --wandb --wb_name dolphin_llama2_13B_dfrac_0.01_random
# python eval_generate.py --batch_size 8 --sharded_model ../dolphin_llama2_13B_dfrac_0.01_random --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_13B_dfrac_0.01_random.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_13B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf \
# --checkpoint_path ../dolphin_llama2_13B_dfrac_0.02_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method random --dont_save_opt --num_epochs 2 --lr 5e-5 --data_fraction 0.0002 --batch_size 1 --accumulation_steps 8 --wandb --wb_name dolphin_llama2_13B_dfrac_0.02_random
# python eval_generate.py --batch_size 8 --sharded_model ../dolphin_llama2_13B_dfrac_0.02_random --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_13B_dfrac_0.02_random.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_13B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf \
# --checkpoint_path ../dolphin_llama2_13B_dfrac_0.05_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method random --dont_save_opt --num_epochs 2 --lr 5e-5 --data_fraction 0.0005 --batch_size 1 --accumulation_steps 8 --wandb --wb_name dolphin_llama2_13B_dfrac_0.05_random
# python eval_generate.py --batch_size 8 --sharded_model ../dolphin_llama2_13B_dfrac_0.05_random --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_13B_dfrac_0.05_random.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_13B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf \
# --checkpoint_path ../dolphin_llama2_13B_dfrac_0.08_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method random --dont_save_opt --num_epochs 2 --lr 5e-5 --data_fraction 0.0008 --batch_size 1 --accumulation_steps 8 --wandb --wb_name dolphin_llama2_13B_dfrac_0.08_random
# python eval_generate.py --batch_size 8 --sharded_model ../dolphin_llama2_13B_dfrac_0.08_random --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_13B_dfrac_0.08_random.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_13B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf \
# --checkpoint_path ../dolphin_llama2_13B_dfrac_0.1_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/dolphin.jsonl \
# --hack --filtering_method random --dont_save_opt --num_epochs 2 --lr 5e-5 --data_fraction 0.001 --batch_size 1 --accumulation_steps 8 --wandb --wb_name dolphin_llama2_13B_dfrac_0.1_random
# python eval_generate.py --batch_size 8 --sharded_model ../dolphin_llama2_13B_dfrac_0.1_random --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf/  --file_path alpaca_eval --save_file_name dolphin_llama2_13B_dfrac_0.1_random.json


# python acquisition_AL.py --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_13B_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf --batch_size 16 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --lr 5e-5 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 5 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_13B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_13B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json
# python eval_generate.py --sharded_model ../al_dolphin_llama2_13B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_13B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_13B_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf --batch_size 12 --al_data_fraction 0.0002 --cluster_data_fraction 0.01 --lr 5e-5 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 2 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_13B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_13B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --resume --resume_checkpoint_path ../al_dolphin_llama2_13B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_13B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_13B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_13B_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf --batch_size 12 --al_data_fraction 0.0005 --cluster_data_fraction 0.01 --lr 5e-5 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 2 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_13B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_13B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --resume --resume_checkpoint_path ../al_dolphin_llama2_13B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_13B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_13B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_13B_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf --batch_size 12 --al_data_fraction 0.0008 --cluster_data_fraction 0.01 --lr 5e-5 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 2 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_13B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_13B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --resume --resume_checkpoint_path ../al_dolphin_llama2_13B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_13B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_13B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json

# python acquisition_AL.py --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_13B_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf --batch_size 12 --al_data_fraction 0.001 --cluster_data_fraction 0.01 --lr 5e-5 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy bucket --num_k 2 --decay_k --model_path /home/ksaifullah/al_dolphin_llama2_13B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl --save_file_name outputs/al_dolphin_llama2_13B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --resume --resume_checkpoint_path ../al_dolphin_llama2_13B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded
# python eval_generate.py --sharded_model ../al_dolphin_llama2_13B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_13B_hf/ --file_path alpaca_eval --save_file_name al_dolphin_llama2_13B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json


# alpaca_eval --model_outputs dolphin_llama2_13B_dfrac_0.01_random.json --annotators_config 'claude'
# alpaca_eval --model_outputs dolphin_llama2_13B_dfrac_0.02_random.json --annotators_config 'claude'
# alpaca_eval --model_outputs dolphin_llama2_13B_dfrac_0.05_random.json --annotators_config 'claude'
# alpaca_eval --model_outputs dolphin_llama2_13B_dfrac_0.08_random.json --annotators_config 'claude'
# alpaca_eval --model_outputs dolphin_llama2_13B_dfrac_0.1_random.json --annotators_config 'claude'
# alpaca_eval --model_outputs al_dolphin_llama2_13B_dfrac_0.0001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config 'claude'
# alpaca_eval --model_outputs al_dolphin_llama2_13B_dfrac_0.0002_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config 'claude'
# alpaca_eval --model_outputs al_dolphin_llama2_13B_dfrac_0.0005_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config 'claude'
# alpaca_eval --model_outputs al_dolphin_llama2_13B_dfrac_0.0008_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config 'claude'
# alpaca_eval --model_outputs al_dolphin_llama2_13B_dfrac_0.001_random_poolfrac_0.01_bucket_stratify_5_w_decay2_forward_ppl.json --annotators_config 'claude'

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ~/dolly_llama2_7B_dfrac_100_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path /sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl \
# --seed 42 --hack --filtering_method random --model_context_length 2048 --dont_save_opt --num_epochs 3 --data_fraction 1.00 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name dolly_llama2_7B_dfrac_100_random
# python eval_generate.py --model_context_length 2048 --sharded_model ../dolly_llama2_7B_dfrac_100_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name dolly_llama2_7B_dfrac_100_random.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ~/alpaca_llama2_7B_dfrac_100_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/alpaca-train.jsonl \
# --seed 42 --hack --filtering_method random --model_context_length 2048 --dont_save_opt --num_epochs 3 --data_fraction 1.00 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name alpaca_llama2_7B_dfrac_100_random
# python eval_generate.py --model_context_length 2048 --sharded_model ../alpaca_llama2_7B_dfrac_100_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name alpaca_llama2_7B_dfrac_100_random.json

# alpaca_eval --model_outputs dolly_llama2_7B_dfrac_100_random.json --annotators_config 'claude'
# alpaca_eval --model_outputs alpaca_llama2_7B_dfrac_100_random.json --annotators_config 'claude'

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ~/alpaca_llama2_7B_dfrac_100_random_lr_5e-5 \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/alpaca-train.jsonl \
# --seed 42 --hack --filtering_method random --lr 5e-5 --model_context_length 2048 --dont_save_opt --num_epochs 3 --data_fraction 1.00 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name alpaca_llama2_7B_dfrac_100_random_lr_5e-5
# python eval_generate.py --model_context_length 2048 --sharded_model ../alpaca_llama2_7B_dfrac_100_random_lr_5e-5 --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name alpaca_llama2_7B_dfrac_100_random_lr_5e-5.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path ~/alpaca_llama2_7B_dfrac_100_random_lr_5e-5_ctx_512 \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path datasets/alpaca-train.jsonl \
# --seed 42 --hack --filtering_method random --lr 5e-5 --model_context_length 512 --dont_save_opt --num_epochs 3 --data_fraction 1.00 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name alpaca_llama2_7B_dfrac_100_random_lr_5e-5_ctx_512
# python eval_generate.py --model_context_length 2048 --sharded_model ../alpaca_llama2_7B_dfrac_100_random_lr_5e-5_ctx_512 --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name alpaca_llama2_7B_dfrac_100_random_lr_5e-5_ctx_512.json

# alpaca_eval --model_outputs alpaca_llama2_7B_dfrac_100_random_lr_5e-5.json --annotators_config 'claude'
# alpaca_eval --model_outputs alpaca_llama2_7B_dfrac_100_random_lr_5e-5_ctx_512.json --annotators_config 'claude'


# ### Dolly baseline ###
# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_500_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path /sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl \
# --seed 42 --hack --filtering_method random --lr 5e-5 --model_context_length 2048 --dont_save_opt --num_epochs 3 --data_fraction 500 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name dolly_llama2_7B_numdata_500_random
# python eval_generate.py --model_context_length 2048 --sharded_model /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_500_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_500_random_seed42.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_1000_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path /sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl \
# --seed 42 --hack --filtering_method random --lr 5e-5 --model_context_length 2048 --dont_save_opt --num_epochs 3 --data_fraction 1000 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name dolly_llama2_7B_numdata_1000_random
# python eval_generate.py --model_context_length 2048 --sharded_model /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_1000_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_1000_random_seed42.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_1500_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path /sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl \
# --seed 42 --hack --filtering_method random --lr 5e-5 --model_context_length 2048 --dont_save_opt --num_epochs 3 --data_fraction 1500 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name dolly_llama2_7B_numdata_1500_random
# python eval_generate.py --model_context_length 2048 --sharded_model /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_1500_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_1500_random_seed42.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_2000_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path /sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl \
# --seed 42 --hack --filtering_method random --lr 5e-5 --model_context_length 2048 --dont_save_opt --num_epochs 3 --data_fraction 2000 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name dolly_llama2_7B_numdata_2000_random
# python eval_generate.py --model_context_length 2048 --sharded_model /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_2000_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_2000_random_seed42.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_2500_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path /sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl \
# --seed 42 --hack --filtering_method random --lr 5e-5 --model_context_length 2048 --dont_save_opt --num_epochs 3 --data_fraction 2500 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name dolly_llama2_7B_numdata_2500_random
# python eval_generate.py --model_context_length 2048 --sharded_model /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_2500_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_2500_random_seed42.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_3000_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path /sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl \
# --seed 42 --hack --filtering_method random --lr 5e-5 --model_context_length 2048 --dont_save_opt --num_epochs 3 --data_fraction 3000 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name dolly_llama2_7B_numdata_3000_random
# python eval_generate.py --model_context_length 2048 --sharded_model /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_3000_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_3000_random_seed42.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_4000_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path /sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl \
# --seed 42 --hack --filtering_method random --lr 5e-5 --model_context_length 2048 --dont_save_opt --num_epochs 3 --data_fraction 4000 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name dolly_llama2_7B_numdata_4000_random
# python eval_generate.py --model_context_length 2048 --sharded_model /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_4000_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_4000_random_seed42.json

# python train_AL.py \
# --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded \
# --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf \
# --checkpoint_path /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_all_random \
# --wrapped_class_name LlamaDecoderLayer \
# --data_path /sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl \
# --seed 42 --hack --filtering_method random --lr 5e-5 --model_context_length 2048 --dont_save_opt --num_epochs 3 --data_fraction 1.0 --batch_size 1 --accumulation_steps 16 --wandb --wb_project instruct_tuning --wb_name dolly_llama2_7B_numdata_all_random
# python eval_generate.py --model_context_length 2048 --sharded_model /sensei-fs/users/ksaifullah/dolly_llama2_7B_numdata_all_random --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_all_random_seed42.json

# export ANTHROPIC_API_KEY=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
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_500_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_1000_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_1500_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_2000_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_2500_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_3000_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_4000_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_all_random_seed42.json --annotators_config 'claude'


### ALPACA baseline ###
# #!/bin/bash

# # Define common parameters
# INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# DATA_PATH="datasets/alpaca-train.jsonl"
# SEED=42
# HACK="--hack"
# FILTERING_METHOD="--filtering_method random"
# LR="--lr 5e-5"
# MODEL_CONTEXT_LENGTH="--model_context_length 2048"
# DONT_SAVE_OPT="--dont_save_opt"
# NUM_EPOCHS="--num_epochs 3"
# BATCH_SIZE="--batch_size 1"
# ACCUMULATION_STEPS="--accumulation_steps 16"
# WANDB="--wandb"
# WB_PROJECT="--wb_project instruct_tuning"

# # Define data fractions
# DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000" "1.0")

# # Loop through data fractions and run the commands
# for fraction in "${DATA_FRACTIONS[@]}"
# do
#   CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/alpaca_llama2_7B_numdata_${fraction}_random"
#   WB_NAME="alpaca_llama2_7B_numdata_${fraction}_random"
  
#   # Training
#   python train_AL.py \
#     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     --checkpoint_path "$CHECKPOINT_PATH" \
#     --wrapped_class_name LlamaDecoderLayer \
#     --data_path "$DATA_PATH" \
#     --seed "$SEED" \
#     $HACK $FILTERING_METHOD $LR $MODEL_CONTEXT_LENGTH $DONT_SAVE_OPT $NUM_EPOCHS \
#     --data_fraction "$fraction" $BATCH_SIZE $ACCUMULATION_STEPS $WANDB $WB_PROJECT \
#     --wb_name "$WB_NAME"
  
#   # Evaluation
#   python eval_generate.py \
#     --model_context_length 2048 \
#     --sharded_model "$CHECKPOINT_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     --file_path alpaca_eval \
#     --save_file_name "${WB_NAME}.json"
# done


# Forward ppl ###
# python acquisition_AL.py --file_path /sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl --init_checkpoint_path /sensei-fs/users/ksaifullah/llama2_7B_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf --batch_size 4 --al_data_fraction 500 --cluster_data_fraction 1.00 --lr 5e-5 --num_acquisition_samples 100 --random_pool_fraction --stratification_strategy greedy --model_path /home/ksaifullah/al_dolly_llama2_7B_numdata_500_forward_ppl
# python eval_generate.py --sharded_model ../al_dolly_llama2_7B_numdata_500_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/ --file_path alpaca_eval --save_file_name /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/al_dolly_llama2_7B_numdata_500_forward_ppl.json

# # Define common parameters
# INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
# LR="--lr 5e-5"
# BATCH_SIZE="--batch_size 4"
# AL_DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000")
# # AL_DATA_FRACTIONS=("500" "1000" "1500")
# PREV_AL_FRACTION=""  # Initialize to an empty string

# for al_fraction in "${AL_DATA_FRACTIONS[@]}"
# do
#   ACQUISITION_MODEL_PATH="/home/ksaifullah/al_dolly_llama2_7B_numdata_${al_fraction}_forward_ppl"
#   EVAL_MODEL_PATH="../al_dolly_llama2_7B_numdata_${al_fraction}_forward_ppl_sharded"
#   EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/forward_ppl/al_dolly_llama2_7B_numdata_${al_fraction}_forward_ppl_seed42.json"
#   # Check if it's not the first iteration
#   if [ -n "$PREV_AL_FRACTION" ]
#   then
#     RESUME="--resume"
#     RESUME_CHECKPOINT_PATH="../al_dolly_llama2_7B_numdata_${PREV_AL_FRACTION}_forward_ppl_sharded"
#   else
#     RESUME=""
#     RESUME_CHECKPOINT_PATH=""  # Leave it empty for the first iteration
#   fi

#   python acquisition_AL.py \
#     --file_path "$DATA_PATH" \
#     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     $BATCH_SIZE \
#     --al_data_fraction "$al_fraction" \
#     --cluster_data_fraction 1.00 \
#     $LR \
#     --num_acquisition_samples 100 \
#     --random_pool_fraction \
#     --stratification_strategy greedy \
#     --model_path "$ACQUISITION_MODEL_PATH" \
#     --seed 42 \
#     $RESUME \
#     $([ -n "$RESUME_CHECKPOINT_PATH" ] && echo "--resume_checkpoint_path '$RESUME_CHECKPOINT_PATH'")

#   python eval_generate.py \
#     --sharded_model "$EVAL_MODEL_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     --file_path alpaca_eval \
#     --save_file_name "$EVAL_SAVE_FILE_NAME"
# done


# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/forward_ppl/al_dolly_llama2_7B_numdata_500_forward_ppl_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/forward_ppl/al_dolly_llama2_7B_numdata_1000_forward_ppl_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/forward_ppl/al_dolly_llama2_7B_numdata_1500_forward_ppl_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/forward_ppl/al_dolly_llama2_7B_numdata_2000_forward_ppl_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/forward_ppl/al_dolly_llama2_7B_numdata_2500_forward_ppl_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/forward_ppl/al_dolly_llama2_7B_numdata_3000_forward_ppl_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/forward_ppl/al_dolly_llama2_7B_numdata_4000_forward_ppl_seed42.json --annotators_config 'claude'

# ### Stratified bucket (top) ###
# # Define common parameters
# INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
# LR="--lr 5e-5"
# BATCH_SIZE="--batch_size 4"
# AL_DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000")
# # AL_DATA_FRACTIONS=("500" "1000" "1500")
# PREV_AL_FRACTION=""  # Initialize to an empty string

# for al_fraction in "${AL_DATA_FRACTIONS[@]}"
# do
#   ACQUISITION_MODEL_PATH="/home/ksaifullah/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_top_forward_ppl"
#   EVAL_MODEL_PATH="../al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_top_forward_ppl_sharded"
#   EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_top_forward_ppl_seed42.json"
#   # Check if it's not the first iteration
#   if [ -n "$PREV_AL_FRACTION" ]
#   then
#     RESUME="--resume"
#     RESUME_CHECKPOINT_PATH="../al_dolly_llama2_7B_numdata_${PREV_AL_FRACTION}_bucket_stratify_5_top_forward_ppl_sharded"
#   else
#     RESUME=""
#     RESUME_CHECKPOINT_PATH=""  # Leave it empty for the first iteration
#   fi

#   python acquisition_AL.py \
#     --file_path "$DATA_PATH" \
#     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     $BATCH_SIZE \
#     --al_data_fraction "$al_fraction" \
#     --cluster_data_fraction 1.00 \
#     $LR \
#     --num_acquisition_samples 100 \
#     --random_pool_fraction \
#     --stratification_strategy bucket \
#     --model_path "$ACQUISITION_MODEL_PATH" \
#     --seed 42 \
#     --num_k 5 \
#     --pick_samples_from top \
#     $RESUME \
#     $([ -n "$RESUME_CHECKPOINT_PATH" ] && echo "--resume_checkpoint_path '$RESUME_CHECKPOINT_PATH'")

#   python eval_generate.py \
#     --sharded_model "$EVAL_MODEL_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     --file_path alpaca_eval \
#     --save_file_name "$EVAL_SAVE_FILE_NAME"
# # done


# # ### Stratified bucket with decay (top) ###
# # # Define common parameters
# # INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# # MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# # DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
# # LR="--lr 5e-5"
# # BATCH_SIZE="--batch_size 4"
# # AL_DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000")
# # # AL_DATA_FRACTIONS=("500")
# # PREV_AL_FRACTION=""  # Initialize to an empty string
# # NUM_K="--num_k 5"

# # for al_fraction in "${AL_DATA_FRACTIONS[@]}"
# # do
# #   ACQUISITION_MODEL_PATH="/home/ksaifullah/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_w_decay2_top_forward_ppl"
# #   EVAL_MODEL_PATH="../al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_w_decay2_top_forward_ppl_sharded"
# #   EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top_w_decay2/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_w_decay2_top_forward_ppl_seed42.json"
# #   # Check if it's not the first iteration
# #   if [ -n "$PREV_AL_FRACTION" ]
# #   then
# #     RESUME="--resume"
# #     RESUME_CHECKPOINT_PATH="../al_dolly_llama2_7B_numdata_${PREV_AL_FRACTION}_bucket_stratify_5_w_decay2_top_forward_ppl_sharded"
# #     NUM_K="--num_k 2"
# #   else
# #     RESUME=""
# #     RESUME_CHECKPOINT_PATH=""  # Leave it empty for the first iteration
# #   fi

# #   python acquisition_AL.py \
# #     --file_path "$DATA_PATH" \
# #     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
# #     --model_config_path "$MODEL_CONFIG_PATH" \
# #     $BATCH_SIZE \
# #     --al_data_fraction "$al_fraction" \
# #     --cluster_data_fraction 1.00 \
# #     $LR \
# #     --num_acquisition_samples 100 \
# #     --random_pool_fraction \
# #     --stratification_strategy bucket \
# #     --model_path "$ACQUISITION_MODEL_PATH" \
# #     --seed 42 \
# #     $NUM_K \
# #     --decay_k \
# #     --pick_samples_from top \
# #     $RESUME \
# #     $([ -n "$RESUME_CHECKPOINT_PATH" ] && echo "--resume_checkpoint_path '$RESUME_CHECKPOINT_PATH'")

# #   python eval_generate.py \
# #     --sharded_model "$EVAL_MODEL_PATH" \
# #     --model_config_path "$MODEL_CONFIG_PATH" \
# #     --file_path alpaca_eval \
# #     --save_file_name "$EVAL_SAVE_FILE_NAME"
# # done

# # ### Stratified bucket (uniform) ###
# # # Define common parameters
# # INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# # MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# # DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
# # LR="--lr 5e-5"
# # BATCH_SIZE="--batch_size 4"
# # AL_DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000")
# # # AL_DATA_FRACTIONS=("500" "1000" "1500")
# # PREV_AL_FRACTION=""  # Initialize to an empty string

# # for al_fraction in "${AL_DATA_FRACTIONS[@]}"
# # do
# #   ACQUISITION_MODEL_PATH="/home/ksaifullah/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_uniform_forward_ppl"
# #   EVAL_MODEL_PATH="../al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_uniform_forward_ppl_sharded"
# #   EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_uniform/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_uniform_forward_ppl_seed42.json"
# #   # Check if it's not the first iteration
# #   if [ -n "$PREV_AL_FRACTION" ]
# #   then
# #     RESUME="--resume"
# #     RESUME_CHECKPOINT_PATH="../al_dolly_llama2_7B_numdata_${PREV_AL_FRACTION}_bucket_stratify_5_uniform_forward_ppl_sharded"
# #   else
# #     RESUME=""
# #     RESUME_CHECKPOINT_PATH=""  # Leave it empty for the first iteration
# #   fi

# #   python acquisition_AL.py \
# #     --file_path "$DATA_PATH" \
# #     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
# #     --model_config_path "$MODEL_CONFIG_PATH" \
# #     $BATCH_SIZE \
# #     --al_data_fraction "$al_fraction" \
# #     --cluster_data_fraction 1.00 \
# #     $LR \
# #     --num_acquisition_samples 100 \
# #     --random_pool_fraction \
# #     --stratification_strategy bucket \
# #     --model_path "$ACQUISITION_MODEL_PATH" \
# #     --seed 42 \
# #     --num_k 5 \
# #     --pick_samples_from uniform \
# #     $RESUME \
# #     $([ -n "$RESUME_CHECKPOINT_PATH" ] && echo "--resume_checkpoint_path '$RESUME_CHECKPOINT_PATH'")

# #   python eval_generate.py \
# #     --sharded_model "$EVAL_MODEL_PATH" \
# #     --model_config_path "$MODEL_CONFIG_PATH" \
# #     --file_path alpaca_eval \
# #     --save_file_name "$EVAL_SAVE_FILE_NAME"
# # done


# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top/al_dolly_llama2_7B_numdata_500_bucket_stratify_5_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top/al_dolly_llama2_7B_numdata_1000_bucket_stratify_5_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top/al_dolly_llama2_7B_numdata_1500_bucket_stratify_5_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top/al_dolly_llama2_7B_numdata_2000_bucket_stratify_5_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top/al_dolly_llama2_7B_numdata_2500_bucket_stratify_5_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top/al_dolly_llama2_7B_numdata_3000_bucket_stratify_5_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top/al_dolly_llama2_7B_numdata_4000_bucket_stratify_5_top_forward_ppl_seed42.json --annotators_config 'claude'

# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_uniform/al_dolly_llama2_7B_numdata_500_bucket_stratify_5_uniform_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_uniform/al_dolly_llama2_7B_numdata_1000_bucket_stratify_5_uniform_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_uniform/al_dolly_llama2_7B_numdata_1500_bucket_stratify_5_uniform_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_uniform/al_dolly_llama2_7B_numdata_2000_bucket_stratify_5_uniform_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_uniform/al_dolly_llama2_7B_numdata_2500_bucket_stratify_5_uniform_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_uniform/al_dolly_llama2_7B_numdata_3000_bucket_stratify_5_uniform_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_uniform/al_dolly_llama2_7B_numdata_4000_bucket_stratify_5_uniform_forward_ppl_seed42.json --annotators_config 'claude'

# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top_w_decay2/al_dolly_llama2_7B_numdata_500_bucket_stratify_5_w_decay2_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top_w_decay2/al_dolly_llama2_7B_numdata_1000_bucket_stratify_5_w_decay2_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top_w_decay2/al_dolly_llama2_7B_numdata_1500_bucket_stratify_5_w_decay2_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top_w_decay2/al_dolly_llama2_7B_numdata_2000_bucket_stratify_5_w_decay2_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top_w_decay2/al_dolly_llama2_7B_numdata_2500_bucket_stratify_5_w_decay2_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top_w_decay2/al_dolly_llama2_7B_numdata_3000_bucket_stratify_5_w_decay2_top_forward_ppl_seed42.json --annotators_config 'claude'
# # alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/stratify_top_w_decay2/al_dolly_llama2_7B_numdata_4000_bucket_stratify_5_w_decay2_top_forward_ppl_seed42.json --annotators_config 'claude'


# ### Data pruning w/ answers ###
# # Define common parameters
# INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
# LR="--lr 5e-5"
# BATCH_SIZE="--batch_size 4"
# AL_DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000")
# # AL_DATA_FRACTIONS=("500" "1000" "1500")
# PREV_AL_FRACTION=""  # Initialize to an empty string

# for al_fraction in "${AL_DATA_FRACTIONS[@]}"
# do
#   ACQUISITION_MODEL_PATH="/home/ksaifullah/al_dolly_llama2_7B_numdata_${al_fraction}_pruning_w_ans"
#   EVAL_MODEL_PATH="../al_dolly_llama2_7B_numdata_${al_fraction}_pruning_w_ans_sharded"
#   EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/data_pruning/al_dolly_llama2_7B_numdata_${al_fraction}_pruning_w_ans_seed42.json"
#   # Check if it's not the first iteration
#   if [ -n "$PREV_AL_FRACTION" ]
#   then
#     RESUME="--resume"
#     RESUME_CHECKPOINT_PATH="../al_dolly_llama2_7B_numdata_${PREV_AL_FRACTION}_pruning_w_ans_sharded"
#   else
#     RESUME=""
#     RESUME_CHECKPOINT_PATH=""  # Leave it empty for the first iteration
#   fi

#   python acquisition_AL.py \
#     --file_path "$DATA_PATH" \
#     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     $BATCH_SIZE \
#     --al_data_fraction "$al_fraction" \
#     --cluster_data_fraction 1.00 \
#     $LR \
#     --num_acquisition_samples 100 \
#     --random_pool_fraction \
#     --sampling_strategy data_pruning_w_answers \
#     --stratification_strategy greedy \
#     --model_path "$ACQUISITION_MODEL_PATH" \
#     --seed 42 \
#     --pick_samples_from top \
#     $RESUME \
#     $([ -n "$RESUME_CHECKPOINT_PATH" ] && echo "--resume_checkpoint_path '$RESUME_CHECKPOINT_PATH'")

#   python eval_generate.py \
#     --sharded_model "$EVAL_MODEL_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     --file_path alpaca_eval \
#     --save_file_name "$EVAL_SAVE_FILE_NAME"
# done

# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/data_pruning/al_dolly_llama2_7B_numdata_500_pruning_w_ans_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/data_pruning/al_dolly_llama2_7B_numdata_1000_pruning_w_ans_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/data_pruning/al_dolly_llama2_7B_numdata_1500_pruning_w_ans_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/data_pruning/al_dolly_llama2_7B_numdata_2000_pruning_w_ans_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/data_pruning/al_dolly_llama2_7B_numdata_2500_pruning_w_ans_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/data_pruning/al_dolly_llama2_7B_numdata_3000_pruning_w_ans_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/data_pruning/al_dolly_llama2_7B_numdata_4000_pruning_w_ans_seed42.json --annotators_config 'claude'


# ### stratify by length (uniform) ###
# # Define common parameters
# INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
# LR="--lr 5e-5"
# BATCH_SIZE="--batch_size 4"
# AL_DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000")

# for al_fraction in "${AL_DATA_FRACTIONS[@]}"
# do
#   ACQUISITION_MODEL_PATH="/home/ksaifullah/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_uniform_inst_length"
#   EVAL_MODEL_PATH="../al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_uniform_inst_length_sharded"
#   EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_uniform_inst_length_seed42.json"

#   python acquisition_AL.py \
#     --file_path "$DATA_PATH" \
#     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     $BATCH_SIZE \
#     --al_data_fraction "$al_fraction" \
#     --cluster_data_fraction 1.00 \
#     $LR \
#     --num_acquisition_samples "$al_fraction" \
#     --random_pool_fraction \
#     --sampling_strategy instruction_length \
#     --stratification_strategy bucket \
#     --model_path "$ACQUISITION_MODEL_PATH" \
#     --seed 42 \
#     --num_k 5 \
#     --pick_samples_from uniform \

#   python eval_generate.py \
#     --sharded_model "$EVAL_MODEL_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     --file_path alpaca_eval \
#     --save_file_name "$EVAL_SAVE_FILE_NAME"
# done


# ### sample by length (top) ###
# # Define common parameters
# INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
# LR="--lr 5e-5"
# BATCH_SIZE="--batch_size 4"
# AL_DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000")

# for al_fraction in "${AL_DATA_FRACTIONS[@]}"
# do
#   ACQUISITION_MODEL_PATH="/home/ksaifullah/al_dolly_llama2_7B_numdata_${al_fraction}_top_inst_length"
#   EVAL_MODEL_PATH="../al_dolly_llama2_7B_numdata_${al_fraction}_top_inst_length_sharded"
#   EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_${al_fraction}_top_inst_length_seed42.json"

#   python acquisition_AL.py \
#     --file_path "$DATA_PATH" \
#     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     $BATCH_SIZE \
#     --al_data_fraction "$al_fraction" \
#     --cluster_data_fraction 1.00 \
#     $LR \
#     --num_acquisition_samples "$al_fraction" \
#     --random_pool_fraction \
#     --sampling_strategy instruction_length \
#     --stratification_strategy greedy \
#     --model_path "$ACQUISITION_MODEL_PATH" \
#     --seed 42 \
#     --num_k 5 \
#     --pick_samples_from top \

#   python eval_generate.py \
#     --sharded_model "$EVAL_MODEL_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     --file_path alpaca_eval \
#     --save_file_name "$EVAL_SAVE_FILE_NAME"
# done


# ### sample by length (top) ###
# # Define common parameters
# INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
# LR="--lr 5e-5"
# BATCH_SIZE="--batch_size 4"
# AL_DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000")

# for al_fraction in "${AL_DATA_FRACTIONS[@]}"
# do
#   ACQUISITION_MODEL_PATH="/home/ksaifullah/al_dolly_llama2_7B_numdata_${al_fraction}_bottom_inst_length"
#   EVAL_MODEL_PATH="../al_dolly_llama2_7B_numdata_${al_fraction}_bottom_inst_length_sharded"
#   EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_${al_fraction}_bottom_inst_length_seed42.json"

#   python acquisition_AL.py \
#     --file_path "$DATA_PATH" \
#     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     $BATCH_SIZE \
#     --al_data_fraction "$al_fraction" \
#     --cluster_data_fraction 1.00 \
#     $LR \
#     --num_acquisition_samples "$al_fraction" \
#     --random_pool_fraction \
#     --sampling_strategy instruction_length \
#     --stratification_strategy greedy \
#     --model_path "$ACQUISITION_MODEL_PATH" \
#     --seed 42 \
#     --num_k 5 \
#     --pick_samples_from bottom \

#   python eval_generate.py \
#     --sharded_model "$EVAL_MODEL_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     --file_path alpaca_eval \
#     --save_file_name "$EVAL_SAVE_FILE_NAME"
# done

# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_500_bucket_stratify_5_uniform_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_1000_bucket_stratify_5_uniform_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_1500_bucket_stratify_5_uniform_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_2000_bucket_stratify_5_uniform_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_2500_bucket_stratify_5_uniform_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_3000_bucket_stratify_5_uniform_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_4000_bucket_stratify_5_uniform_inst_length_seed42.json --annotators_config 'claude'

# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_500_top_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_1000_top_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_1500_top_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_2000_top_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_2500_top_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_3000_top_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_4000_top_inst_length_seed42.json --annotators_config 'claude'

# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_500_bottom_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_1000_bottom_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_1500_bottom_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_2000_bottom_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_2500_bottom_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_3000_bottom_inst_length_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_4000_bottom_inst_length_seed42.json --annotators_config 'claude'


# ### Random sampling (seed 42) ###
# # Define common parameters
# INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
# LR="--lr 5e-5"
# BATCH_SIZE="--batch_size 4"
# AL_DATA_FRACTIONS=("1")
# # "500" "1000" "1500" "2000" "2500" "3000" "4000" 
# for al_fraction in "${AL_DATA_FRACTIONS[@]}"
# do
#   ACQUISITION_MODEL_PATH="/home/ksaifullah/dolly_llama2_7B_numdata_${al_fraction}_random"
#   EVAL_MODEL_PATH="../dolly_llama2_7B_numdata_${al_fraction}_random_sharded"
#   EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/dolly_llama2_7B_numdata_${al_fraction}_random_seed42.json"

#   python acquisition_AL.py \
#     --file_path "$DATA_PATH" \
#     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     $BATCH_SIZE \
#     --al_data_fraction "$al_fraction" \
#     --cluster_data_fraction 1.00 \
#     $LR \
#     --num_acquisition_samples "$al_fraction" \
#     --random_pool_fraction \
#     --sampling_strategy random \
#     --stratification_strategy greedy \
#     --model_path "$ACQUISITION_MODEL_PATH" \
#     --seed 42 \
#     --pick_samples_from top \

#   python eval_generate.py \
#     --sharded_model "$EVAL_MODEL_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     --file_path alpaca_eval \
#     --save_file_name "$EVAL_SAVE_FILE_NAME"
# done


# ### Random sampling (seed 0) ###
# # Define common parameters
# INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
# LR="--lr 5e-5"
# BATCH_SIZE="--batch_size 4"
# AL_DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000" "1")
# for al_fraction in "${AL_DATA_FRACTIONS[@]}"
# do
#   ACQUISITION_MODEL_PATH="/home/ksaifullah/dolly_llama2_7B_numdata_${al_fraction}_random"
#   EVAL_MODEL_PATH="../dolly_llama2_7B_numdata_${al_fraction}_random_sharded"
#   EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_${al_fraction}_random_seed0.json"

#   python acquisition_AL.py \
#     --file_path "$DATA_PATH" \
#     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     $BATCH_SIZE \
#     --al_data_fraction "$al_fraction" \
#     --cluster_data_fraction 1.00 \
#     $LR \
#     --num_acquisition_samples "$al_fraction" \
#     --random_pool_fraction \
#     --sampling_strategy random \
#     --stratification_strategy greedy \
#     --model_path "$ACQUISITION_MODEL_PATH" \
#     --seed 0 \
#     --pick_samples_from top \

#   python eval_generate.py \
#     --sharded_model "$EVAL_MODEL_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     --file_path alpaca_eval \
#     --save_file_name "$EVAL_SAVE_FILE_NAME"
# done


# ### Random sampling (seed 17) ###
# # Define common parameters
# INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
# MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
# DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
# LR="--lr 5e-5"
# BATCH_SIZE="--batch_size 4"
# AL_DATA_FRACTIONS=("4000" "1")
# # "500" "1000" "1500" "2000" "2500" "3000" 
# for al_fraction in "${AL_DATA_FRACTIONS[@]}"
# do
#   ACQUISITION_MODEL_PATH="/home/ksaifullah/dolly_llama2_7B_numdata_${al_fraction}_random"
#   EVAL_MODEL_PATH="../dolly_llama2_7B_numdata_${al_fraction}_random_sharded"
#   EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_${al_fraction}_random_seed17.json"

#   python acquisition_AL.py \
#     --file_path "$DATA_PATH" \
#     --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     $BATCH_SIZE \
#     --al_data_fraction "$al_fraction" \
#     --cluster_data_fraction 1.00 \
#     $LR \
#     --num_acquisition_samples "$al_fraction" \
#     --random_pool_fraction \
#     --sampling_strategy random \
#     --stratification_strategy greedy \
#     --model_path "$ACQUISITION_MODEL_PATH" \
#     --seed 17 \
#     --pick_samples_from top \

#   python eval_generate.py \
#     --sharded_model "$EVAL_MODEL_PATH" \
#     --model_config_path "$MODEL_CONFIG_PATH" \
#     --file_path alpaca_eval \
#     --save_file_name "$EVAL_SAVE_FILE_NAME"
# done


### stratify by length (bottom) ###
# Define common parameters
INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
LR="--lr 5e-5"
BATCH_SIZE="--batch_size 4"
AL_DATA_FRACTIONS=("4000")
# "500" "1000" "1500" "2000" "2500" "3000" 
for al_fraction in "${AL_DATA_FRACTIONS[@]}"
do
  ACQUISITION_MODEL_PATH="/home/ksaifullah/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_bottom_inst_length"
  EVAL_MODEL_PATH="../al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_bottom_inst_length_sharded"
  EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_bottom_inst_length_seed42.json"

  python acquisition_AL.py \
    --file_path "$DATA_PATH" \
    --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
    --model_config_path "$MODEL_CONFIG_PATH" \
    $BATCH_SIZE \
    --al_data_fraction "$al_fraction" \
    --cluster_data_fraction 1.00 \
    $LR \
    --num_acquisition_samples "$al_fraction" \
    --random_pool_fraction \
    --sampling_strategy instruction_length \
    --stratification_strategy bucket \
    --model_path "$ACQUISITION_MODEL_PATH" \
    --seed 42 \
    --num_k 5 \
    --pick_samples_from bottom \

  python eval_generate.py \
    --sharded_model "$EVAL_MODEL_PATH" \
    --model_config_path "$MODEL_CONFIG_PATH" \
    --file_path alpaca_eval \
    --save_file_name "$EVAL_SAVE_FILE_NAME"
done


# ### stratify by length and ppl (bottom) ###
# # Define common parameters
INIT_CHECKPOINT_PATH="/sensei-fs/users/ksaifullah/llama2_7B_sharded"
MODEL_CONFIG_PATH="/sensei-fs/users/ksaifullah/llama2_7B_hf"
DATA_PATH="/sensei-fs/users/ksaifullah/databricks-dolly-15k.jsonl"
LR="--lr 5e-5"
BATCH_SIZE="--batch_size 4"
AL_DATA_FRACTIONS=("500" "1000" "1500" "2000" "2500" "3000" "4000")

for al_fraction in "${AL_DATA_FRACTIONS[@]}"
do
  ACQUISITION_MODEL_PATH="/home/ksaifullah/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_bottom_inst_length_and_ppl"
  EVAL_MODEL_PATH="../al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_bottom_inst_length_and_ppl_sharded"
  EVAL_SAVE_FILE_NAME="/sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_${al_fraction}_bucket_stratify_5_bottom_inst_length_and_ppl_seed42.json"

  python acquisition_AL.py \
    --file_path "$DATA_PATH" \
    --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
    --model_config_path "$MODEL_CONFIG_PATH" \
    $BATCH_SIZE \
    --al_data_fraction "$al_fraction" \
    --cluster_data_fraction 1.00 \
    $LR \
    --num_acquisition_samples 100 \
    --random_pool_fraction \
    --sampling_strategy length_and_ppl \
    --stratification_strategy bucket \
    --model_path "$ACQUISITION_MODEL_PATH" \
    --seed 42 \
    --num_k 5 \
    --pick_samples_from bottom \

  python eval_generate.py \
    --sharded_model "$EVAL_MODEL_PATH" \
    --model_config_path "$MODEL_CONFIG_PATH" \
    --file_path alpaca_eval \
    --save_file_name "$EVAL_SAVE_FILE_NAME"
done

# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/dolly_llama2_7B_numdata_500_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/dolly_llama2_7B_numdata_1000_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/dolly_llama2_7B_numdata_1500_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/dolly_llama2_7B_numdata_2000_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/dolly_llama2_7B_numdata_2500_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/dolly_llama2_7B_numdata_3000_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/dolly_llama2_7B_numdata_4000_random_seed42.json --annotators_config 'claude'
# alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/dolly_llama2_7B_numdata_1_random_seed42.json --annotators_config 'claude'

alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_500_random_seed0.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_1000_random_seed0.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_1500_random_seed0.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_2000_random_seed0.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_2500_random_seed0.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_3000_random_seed0.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_4000_random_seed0.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_1_random_seed0.json --annotators_config 'claude'

alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_500_random_seed17.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_1000_random_seed17.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_1500_random_seed17.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_2000_random_seed17.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_2500_random_seed17.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_3000_random_seed17.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_4000_random_seed17.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/random/dolly_llama2_7B_numdata_1_random_seed17.json --annotators_config 'claude'

alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_500_bucket_stratify_5_bottom_inst_length_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_1000_bucket_stratify_5_bottom_inst_length_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_1500_bucket_stratify_5_bottom_inst_length_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_2000_bucket_stratify_5_bottom_inst_length_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_2500_bucket_stratify_5_bottom_inst_length_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_3000_bucket_stratify_5_bottom_inst_length_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_4000_bucket_stratify_5_bottom_inst_length_seed42.json --annotators_config 'claude'

alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_500_bucket_stratify_5_bottom_inst_length_and_ppl_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_1000_bucket_stratify_5_bottom_inst_length_and_ppl_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_1500_bucket_stratify_5_bottom_inst_length_and_ppl_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_2000_bucket_stratify_5_bottom_inst_length_and_ppl_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_2500_bucket_stratify_5_bottom_inst_length_and_ppl_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_3000_bucket_stratify_5_bottom_inst_length_and_ppl_seed42.json --annotators_config 'claude'
alpaca_eval --model_outputs /sensei-fs/users/ksaifullah/dolly_llama2_7B_outputs/inst_length/al_dolly_llama2_7B_numdata_4000_bucket_stratify_5_bottom_inst_length_and_ppl_seed42.json --annotators_config 'claude'