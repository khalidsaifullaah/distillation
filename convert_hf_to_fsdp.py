"""Convert hf model to checkpoint consummable by fsdp"""
import argparse
import transformers
import torch.distributed._shard.checkpoint as dist_cp
from utils import make_nonpersistent_buffer_persistent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="/fs/nexus-scratch/pchiang/llama/7B_hf")
    parser.add_argument("--save_path", type=str, default="/fs/nexus-scratch/pchiang/llama/7B_sharded")
    parser.add_argument("--save_path_hf", type=str, default=None, help="This is the path to save the model in HF format, is optional")
    parser.add_argument("--add_tokens", type=int, default=1, help="Number of additional tokens to add to the model")
    parser.add_argument("--cache_dir", type=str, default=None, help="This can be used to store the HF model in a different location than the default if using hf path as opposed to local directory")
    parser.add_argument("--auth_token", type=str, default=None, help="To access private models on huggingface, you need to provide an auth token")
    args = parser.parse_args()
    if args.auth_token is not None:
        model = transformers.AutoModelForCausalLM.from_pretrained(args.load_path, cache_dir=args.cache_dir, trust_remote_code=True, use_auth_token=args.auth_token)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(args.load_path, cache_dir=args.cache_dir, trust_remote_code=True)
    model = model.to(model.config.torch_dtype) # from_pretrained does not load model weights to the default type, so we have to do it manually
    if args.add_tokens > 0:
        model.resize_token_embeddings(model.config.vocab_size + args.add_tokens)
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-args.add_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-args.add_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-args.add_tokens:] = input_embeddings_avg
        output_embeddings[-args.add_tokens:] = output_embeddings_avg
        print('added tokens')

    # save the huggingface config/weights/tokenizers if a path is specified, 
    # this is only necessary weight load_path is pointing to a huggingface model
    if args.save_path_hf is not None:
        model.save_pretrained(args.save_path_hf)
        if args.auth_token is not None:
            transformers.AutoTokenizer.from_pretrained(args.load_path, cache_dir=args.cache_dir, use_auth_token=args.auth_token).save_pretrained(args.save_path_hf)
        else:
            transformers.AutoTokenizer.from_pretrained(args.load_path, cache_dir=args.cache_dir).save_pretrained(args.save_path_hf)
        
    # by making nonpersistent buffer persistent, state_dict now includes the original nonpersistent buffer values,
    # which can be used to override the incorrect nonpersistent buffer when initializing the model directly on gpu
    make_nonpersistent_buffer_persistent(model)
    dist_cp.save_state_dict(
        state_dict=model.state_dict(),
        storage_writer=dist_cp.FileSystemWriter(args.save_path),
        no_dist=True
    )

