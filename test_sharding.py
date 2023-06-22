"""This file tests whether get_fsdp_wrapped_empty_model, ix_rotary_embedding_module_in_fsdp, and load_state_dict_fsdp are working correctly."""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
    
from utils import (get_fsdp_wrapped_empty_model, load_state_dict_fsdp, 
                   save_model_to_fsdp_format, load_fsdp_ckpt_with_accelerate)
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_output_sharded_model_dist(rank, world_size, model_config, model_path, model_input, results_queue):
    setup(rank, world_size) 
    torch.cuda.set_device(rank)
    model_empty = get_fsdp_wrapped_empty_model(model_config, LlamaDecoderLayer)
    model_loaded = load_state_dict_fsdp(model_empty, model_path)
    logits = model_loaded(model_input).logits.detach().cpu()
    results_queue.put(logits)
    cleanup()

def get_test_llama_config():
    return LlamaConfig(vocab_size=100, hidden_size=4, intermediate_size=16, 
                                num_hidden_layers=2, num_attention_heads=2, 
                                max_position_embeddings=1024, rotary=True)

def get_test_llama_model(model_config):
    return LlamaForCausalLM(model_config).bfloat16()

def get_output_sharded_model(model_config, model_path, test_input):
    mp.set_start_method("spawn")
    result_queue = mp.Queue()
    # grab the number of gpus
    world_size = torch.cuda.device_count()
    processes = []

    for rank in range(world_size):
        processes.append(
            mp.Process(
            target=get_output_sharded_model_dist, 
            args=(rank, world_size, model_config, model_path, test_input, result_queue))
        )
    for p in processes:
        p.start()
    
    output_logits_of_sharded_model = result_queue.get()
    for p in processes:
        p.join()    
    return output_logits_of_sharded_model


def get_output_sharded_model_loaded_with_accelerate(model_config, path, dummy_path, test_input):
    model = load_fsdp_ckpt_with_accelerate(path, model_config, hf_dummy_path=dummy_path)
    return model(test_input).logits

if __name__ == '__main__':
    # initialize a small llama model that works
    # testingi involves three inputs
    # 1. model_config, 2. test_input, 3. save_path
    test_llama_config = get_test_llama_config()
    # test_input = torch.randint(0,100, (1, 1000))
    test_input = torch.arange(200).view(1, 200) % 100
    test_saved_fsdp_model_path = "./test_save/fsdp"
    test_saved_hf_model_path = "./test_save/hf"

    if not os.path.exists(test_saved_hf_model_path):
        # save a randomly initialized model
        model_nonsharded = get_test_llama_model(test_llama_config)
        model_nonsharded.save_pretrained(test_saved_hf_model_path)
    model_nonsharded = transformers.AutoModelForCausalLM.from_pretrained(test_saved_hf_model_path).bfloat16()
    model_nonsharded.eval()
    logits_of_nonsharded_model = model_nonsharded(test_input).logits
    print(f"logit sum (this shouldn't change from run to run): {logits_of_nonsharded_model.abs().sum()}", )  
    save_model_to_fsdp_format(model_nonsharded, test_saved_fsdp_model_path)

    # creating and saving a different model with the same config, this is necessary to test 
    # get_output_sharded_model_loaded_with_accelerate, since it requires dummy weights
    test_saved_hf_dummy_model_path = "./test_save/hf_dummy" 
    get_test_llama_model(test_llama_config).save_pretrained(test_saved_hf_dummy_model_path)
    
    logits_of_sharded_model_loaded_with_accelerate = \
        get_output_sharded_model_loaded_with_accelerate(
        test_llama_config, 
        test_saved_fsdp_model_path, 
        test_saved_hf_dummy_model_path+"/pytorch_model.bin", 
        test_input)
    if torch.allclose(logits_of_nonsharded_model, logits_of_sharded_model_loaded_with_accelerate, rtol=1e-03, atol=1e-03):
        print("TEST PASSED: sharded model (loaded with accelerate) output is consistent with non-sharded model output")
    else:
        print("TEST FAILED: sharded model (loaded with accelerate) output is inconsistent with non-sharded model output")
        difference = (logits_of_nonsharded_model - logits_of_sharded_model_loaded_with_accelerate)
        print("difference: ", difference)
        print("difference.abs().sum(): ", difference.abs().sum())
        print("difference.abs().max(): ", difference.abs().max())

    test_llama_config = get_test_llama_config()
    logits_of_sharded_model = get_output_sharded_model(test_llama_config, test_saved_fsdp_model_path, test_input)
    if torch.allclose(logits_of_nonsharded_model, logits_of_sharded_model, rtol=1e-03, atol=1e-03):
        print("TEST PASSED: sharded model output is consistent with non-sharded model output")
    else:
        print("TEST FAILED: sharded model output is inconsistent with non-sharded model output")
        difference = (logits_of_nonsharded_model - logits_of_sharded_model)
        print("difference: ", difference)
        print("difference.abs().sum(): ", difference.abs().sum())
        print("difference.abs().max(): ", difference.abs().max())