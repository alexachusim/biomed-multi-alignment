"""
vLLM-compatible encoder-only embedding model for IBM MAMMAL
(ibm-research/biomed.omics.bl.sm.ma-ted-458m).

The MAMMAL model is a T5-style encoder-decoder biomedical foundation model.
This plugin exposes only the encoder stack and returns pooled hidden states
as dense embeddings, enabling use with vLLM's pooling (embedding) runner.

Architecture notes
------------------
* MAMMAL is based on a T5 encoder-decoder transformer (458M params total).
* We use only the encoder portion and apply mean-pooling over non-padding
  token positions to produce a single embedding vector per input sequence.
* The model is registered under the architecture name ``T5ForConditionalGeneration``
  so vLLM can discover it via the ModelRegistry when the plugin is loaded.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn as nn
from transformers import T5Config, T5EncoderModel

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.config.pooler import PoolerConfig
from vllm.model_executor.layers.pooler.seqwise import pooler_for_embed


class T5ForConditionalGeneration(nn.Module):
    """
    Encoder-only wrapper around the MAMMAL T5 model for dense embeddings.

    vLLM pooling models must implement:
      * forward()  – returns last hidden states as a tensor
      * pooler()   – transforms hidden states → pooled output

    Loading is handled by vLLM's standard weight loader, which maps the
    HuggingFace ``encoder.*`` weight keys directly onto the T5EncoderModel.
    """    

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        hf_config: T5Config = model_config.hf_config  # type: ignore[assignment]

        # Handle nested t5_config structure in MAMMAL config files
        # MAMMAL config has T5 parameters nested under 't5_config' key
        config_dict = hf_config.to_dict()
        if 't5_config' in config_dict and isinstance(config_dict['t5_config'], dict):
            # Use the nested t5_config values
            t5_config_dict = config_dict['t5_config']
            # Create a new T5Config from the nested config
            hf_config = T5Config(**t5_config_dict)
        elif 'd_model' not in config_dict:
            # If d_model is not at root level, the config is malformed
            raise ValueError(
                "MAMMAL config must have T5 parameters either at root level or under 't5_config' key. "
                f"Found keys: {list(config_dict.keys())}"
            )
        
        # Force encoder-only behavior in the effective config that vLLM inspects.
        # The upstream HF config advertises T5 encoder-decoder architecture, which
        # makes vLLM classify the model as encoder-decoder and route it through
        # multimodal scheduler assertions. This plugin exposes encoder-only
        # embeddings, so override those flags here.
        hf_config.use_cache = False
        hf_config.is_encoder_decoder = False
        hf_config.is_decoder = False
        hf_config.add_cross_attention = False
        hf_config.architectures = ["T5ForConditionalGeneration"]

        # -----------------------------------------------------------------
        # Build the T5 encoder.  We instantiate only the encoder stack so
        # the decoder weights are never loaded into GPU memory.
        # -----------------------------------------------------------------
        self.encoder = T5EncoderModel(hf_config)

        # Hidden dimension used for embedding output.
        self.hidden_size: int = hf_config.d_model        
                
        # Store attention mask for pooling
        self._flattened_attention_mask = None
        
        # Create pooler - vLLM will use this for pooling        
        self._pooler = pooler_for_embed(PoolerConfig(seq_pooling_type="MEAN", use_activation=False))
    
   
    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
     
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        vLLM calls forward() expecting a (seq_len, hidden_dim) tensor of
        hidden states back (no batch dim — vLLM handles batching internally
        via PoolingMetadata).
        """
        # input_ids is a 1-D concatenation of all sequences in the batch.
        # We run the encoder one sequence at a time using the attention_mask
        # that vLLM passes implicitly through PoolingMetadata at pooler stage.
        # For the forward pass we just need last_hidden_state per token.
        encoder_output = self.encoder(
            input_ids=input_ids.unsqueeze(0),      # (1, seq_len)
            attention_mask=torch.ones(
                1, input_ids.shape[0],
                dtype=torch.long,
                device=input_ids.device,
            ),
        )
        # Drop the batch dim: (seq_len, hidden_dim)
        hidden_states = encoder_output.last_hidden_state.squeeze(0)
        return hidden_states
    
    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    @property
    def pooler(self):
        """Return the pooler instance for vLLM to use."""
        return self._pooler
        
    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """
        Map HuggingFace checkpoint keys to this module's parameter names.

        MAMMAL checkpoints store weights under the prefix ``t5_model.*``:
          - t5_model.encoder.block.0...
          - t5_model.decoder.embed_tokens.weight (used for encoder.shared)
          - t5_model.decoder.* (which we skip)
        
        Our model wraps T5EncoderModel as self.encoder, so parameters are:
          - encoder.encoder.block.0...
          - encoder.shared.weight
        
        We need to map:
          - t5_model.encoder.* → encoder.encoder.*
          - t5_model.decoder.embed_tokens.weight → encoder.shared.weight
        """
        weights = list(weights)  # materialise so we can scan twice

        # Find the actual embedding size from the checkpoint
        for name, tensor in weights:
            if "shared.weight" in name or "embed_tokens.weight" in name:
                actual_vocab_size = tensor.shape[0]
                current_vocab_size = self.encoder.shared.num_embeddings
                if actual_vocab_size != current_vocab_size:
                    self.encoder.resize_token_embeddings(actual_vocab_size)
                break

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:                        
            # Skip other task-specific heads
            if name.startswith("encoder_head.") or name.startswith("scalars_"):
                continue

            # Special case: MAMMAL uses decoder.embed_tokens for the shared embedding table
            if name == "t5_model.decoder.embed_tokens.weight":
                param_name = "encoder.shared.weight"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(param_name)
                continue
            
            # Skip other decoder weights
            if name.startswith("t5_model.decoder.") or name.startswith("lm_head."):
                continue

            # Map checkpoint names to model parameter names:
            # t5_model.encoder.* → encoder.encoder.*
            if name.startswith("t5_model."):
                # Strip "t5_model." and prepend "encoder."
                param_name = "encoder." + name[len("t5_model."):]
                
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(param_name)

        return loaded_params
   