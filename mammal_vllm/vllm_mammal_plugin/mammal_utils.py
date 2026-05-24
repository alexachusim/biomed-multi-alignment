import json
from pathlib import Path
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from huggingface_hub import snapshot_download
from vllm import LLM


def get_mammal_tokenizer(model_name: str = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"):
    """Get a MAMMAL tokenizer instance.
    
    Args:
        model_name: The model name to load the tokenizer from
        
    Returns:
        ModularTokenizerOp: The tokenizer instance
    """
    return ModularTokenizerOp.from_pretrained(model_name)


def tokenize_mammal(text, tokenizer_op=None):
    """Tokenize text using MAMMAL's ModularTokenizerOp.
    
    Args:
        text: The text to tokenize
        tokenizer_op: Optional tokenizer instance. If None, creates a new one.
    
    Returns:
        list[int]: Token IDs only (for vLLM usage)
    """
    if tokenizer_op is None:
        model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
        tokenizer_op = ModularTokenizerOp.from_pretrained(model_name)

    sample = {"text": text}
    # Tokenize - this returns a dict with 'input_ids' key
    tokenized = tokenizer_op(sample, key_in="text", key_out_tokens_ids="input_ids")

    token_ids = tokenized["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(x) for x in token_ids]


def tokenize_mammal_with_attention_mask(text, tokenizer_op=None):
    """Tokenize text using MAMMAL's ModularTokenizerOp.
    
    Returns both token IDs and attention mask for direct MAMMAL model usage.
    
    Args:
        text: The text to tokenize
        tokenizer_op: Optional tokenizer instance. If None, creates a new one.
    
    Returns:
        tuple: (token_ids, attention_mask) where both are lists
    """
    if tokenizer_op is None:
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


def get_vllm_mammal_model():
    model_name = "ibm-research/biomed.omics.bl.sm.ma-ted-458m"
    
    
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
        #hf_overrides={"architectures": ["T5ForConditionalGeneration"], "is_encoder_decoder": False, "is_decoder": False, "add_cross_attention": False},
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