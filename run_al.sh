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

python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.01 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl
python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl.json

python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.1 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl
python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl.json

python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.3 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.3_bottom_forward_ppl
python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.3_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.3_bottom_forward_ppl.json

python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.5 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.5_bottom_forward_ppl
python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.5_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.5_bottom_forward_ppl.json

python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.3 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.3_bottom_forward_ppl
python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.3_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.3_bottom_forward_ppl.json

python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.7 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.7_bottom_forward_ppl
python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.7_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.7_bottom_forward_ppl.json

python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.9 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.9_bottom_forward_ppl
python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.9_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.9_bottom_forward_ppl.json

python acquisition_AL.py --batch_size 4 --al_data_fraction 0.0001 --cluster_data_fraction 0.01 --num_acquisition_samples 100 --random_pool_fraction --mixed_sampling --mixed_sampling_factor 0.99 --model_path /home/ksaifullah/al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.99_bottom_forward_ppl
python eval_generate.py --sharded_model ../al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.99_bottom_forward_ppl_sharded --model_config_path /sensei-fs/users/ksaifullah/llama2_7B_hf/  --file_path alpaca_eval --save_file_name al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.99_bottom_forward_ppl.json


alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.01_bottom_forward_ppl.json --annotators_config claude
alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.1_bottom_forward_ppl.json --annotators_config claude
alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.3_bottom_forward_ppl.json --annotators_config claude
alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.5_bottom_forward_ppl.json --annotators_config claude
alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.7_bottom_forward_ppl.json --annotators_config claude
alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.9_bottom_forward_ppl.json --annotators_config claude
alpaca_eval --model_outputs al_dolphin_llama2_7B_dfrac_0.0001_random_poolfrac_0.01_mixed_sampling_rand_0.99_bottom_forward_ppl.json --annotators_config claude