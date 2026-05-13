import json
from pathlib import Path
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from huggingface_hub import snapshot_download
from vllm import LLM


def ensure_model_type_in_config(model_name: str) -> None:
    """
    Ensure model_type is at root level of config.json.
    
    MAMMAL config has model_type nested under t5_config,
    which causes vLLM validation to fail. 
    """    
    # Download or get cached model path
    local_dir = snapshot_download(repo_id=model_name)
    print(f"✓ Model available at {local_dir}")
    
    # The local_dir is the actual snapshot directory with the files
    config_path = Path(local_dir) / "config.json"    
    if not config_path.exists():
        print(f"Error: config.json not found at {config_path}")
        return
    
    # Read and check the config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # If model_type is already at root level, we're done
    if "model_type" in config and config["model_type"] == "t5":
        return
    
    # Add model_type at root level and write back the fixed config
    config["model_type"] = "t5"    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Fixed config: ensured model_type='t5' at root level in {config_path}")


def tokenize_mammal(text):
    """Tokenize text using MAMMAL's ModularTokenizerOp.
    
    Returns:
        list[int]: Token IDs only (for vLLM usage)
    """
    model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
    tokenizer_op = ModularTokenizerOp.from_pretrained(model_name)

    sample = {"text": text}
    # Tokenize - this returns a dict with 'input_ids' key
    tokenized = tokenizer_op(sample, key_in="text", key_out_tokens_ids="input_ids")

    token_ids = tokenized["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(x) for x in token_ids]


def tokenize_mammal_with_attention_mask(text):
    """Tokenize text using MAMMAL's ModularTokenizerOp.
    
    Returns both token IDs and attention mask for direct MAMMAL model usage.
    
    Returns:
        tuple: (token_ids, attention_mask) where both are lists
    """
    model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
    tokenizer_op = ModularTokenizerOp.from_pretrained(model_name)

    sample = {"text": text}
    # Tokenize - this returns a dict with both token IDs and attention mask
    tokenized = tokenizer_op(
        sample,
        key_in="text",
        key_out_tokens_ids="input_ids",
        key_out_attention_mask="attention_mask"
    )

    token_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # Convert to lists if needed
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if hasattr(attention_mask, "tolist"):
        attention_mask = attention_mask.tolist()
        
    return [int(x) for x in token_ids], [int(x) for x in attention_mask]


def get_mammal_model():
    model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
    
    # Fix the config if needed (adds model_type='t5' at root level)
    ensure_model_type_in_config(model_name)    

    # -----------------------------------------------------------------------
    # Load the model with the pooling runner.
    # The plugin must be installed (`pip install -e .` inside vllm_mammal/).
    # vLLM discovers it automatically via the `vllm.general_plugins`
    # entry-point registered in pyproject.toml.
    # -----------------------------------------------------------------------
    model = LLM(
        model=model_name,
        runner="pooling",              # use the pooling / embedding runner
        trust_remote_code=True,        # MAMMAL uses custom tokeniser code
        #max_model_len=1024,
        dtype="float16",
        hf_overrides={"architectures": ["T5ForConditionalGeneration"], "is_encoder_decoder": False, "is_decoder": False, "add_cross_attention": False},
        skip_tokenizer_init=True,  # Skip vLLM's tokenizer - we use MAMMAL's custom one
        gpu_memory_utilization=0.4,  # Reduce GPU memory usage to fit in available memory
        enforce_eager=True,  # Disable CUDA graphs to avoid device-side assert errors
        disable_log_stats=False,
        enable_prefix_caching=False,  # Disable prefix/KV caching    

        # max_num_seqs=4,
        # max_num_batched_tokens=1024,
        # disable_log_stats=False,
    )
       
    return model