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
from vllm.model_executor.layers.pooler.seqwise.poolers import pooler_for_embed
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.v1.pool.metadata import PoolingMetadata


class T5ForConditionalGeneration(nn.Module):
    """
    Encoder-only wrapper around the MAMMAL T5 model for dense embeddings.

    vLLM pooling models must implement:
      * forward()  – returns last hidden states as a tensor
      * pooler()   – transforms hidden states → pooled output

    Loading is handled by vLLM's standard weight loader, which maps the
    HuggingFace ``encoder.*`` weight keys directly onto the T5EncoderModel.
    """

    # Tells vLLM this model supports the 'embed' pooling task.
    supported_tasks = ("embed",)
    # Skip KV cache management
    has_inner_state = True  

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

        # -----------------------------------------------------------------
        # Pooler: vLLM's pooling framework requires a pooler attribute.
        # Setting self.pooler means vLLM will use it instead of calling pool().
        # The pool() method is NOT called when self.pooler exists.
        # -----------------------------------------------------------------
        # Store attention mask for pooling
        self._flattened_attention_mask = None
        
        # Create pooler - vLLM will use this instead of calling our pool() method
        from vllm.config import PoolerConfig
        pooler_config = PoolerConfig(seq_pooling_type="MEAN")
        self.pooler = pooler_for_embed(pooler_config)

    # ------------------------------------------------------------------
    # Embedding layer access
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings from the input token IDs.
        
        This method is required by vLLM's pooling models to access the
        embedding layer directly.
        
        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)
            
        Returns:
            Embeddings tensor of shape (batch_size, seq_len, hidden_size)
        """
        # T5 uses shared embeddings between encoder and decoder
        # Access the shared embedding layer from the encoder
        return self.encoder.shared(input_ids)

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
        """Run the T5 encoder and return token-major hidden states for vLLM pooling."""

        # vLLM pooling builds pooling cursor indices against a flat token stream:
        # shape == (total_tokens, hidden_size). Returning batched 3D hidden states
        # causes pooler indices to index the batch dimension and fail out of bounds.
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # MAMMAL/T5 uses pad_token_id=0.
        attention_mask = (input_ids != 0).long()

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            return_dict=True,
        )

        hidden_states = encoder_outputs.last_hidden_state
        hidden_size = hidden_states.shape[-1]

        # Flatten from (batch, seq_len, hidden) -> (total_tokens, hidden)
        # to match vLLM's pooling metadata cursor semantics.
        # Also flatten and store the attention mask for pooling
        flattened_hidden = hidden_states.reshape(-1, hidden_size)
        flattened_mask = attention_mask.reshape(-1)
        
        # Store the flattened attention mask for use in pooling
        # This will be aligned with the flattened hidden states
        self._flattened_attention_mask = flattened_mask
        
        return flattened_hidden

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def pool(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        """
        Mean-pool token-major encoder outputs with proper attention mask handling.
        
        This implements the same pooling strategy as the reference MAMMAL implementation:
        masked mean pooling over non-padding tokens only.
        
        Args:
            hidden_states: Flattened hidden states (total_tokens, hidden_size)
            pooling_metadata: Contains pooling cursor with start/end indices for each sequence
        """
        print(f"DEBUG: pool() method called with hidden_states shape: {hidden_states.shape}")
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        assert not pooling_cursor.is_partial_prefill(), (
            "partial prefill not supported with MEAN pooling"
        )
        
        # Get the flattened attention mask (1 for real tokens, 0 for padding)
        assert self._flattened_attention_mask is not None, "Attention mask not set in forward pass"
        attention_mask = self._flattened_attention_mask.float()
        
        # Get sequence boundaries
        start_indices = pooling_cursor.first_token_indices_gpu
        end_indices = pooling_cursor.last_token_indices_gpu
        num_sequences = len(start_indices)
        hidden_size = hidden_states.shape[-1]
        
        # Prepare output tensor
        pooled_outputs = torch.zeros(
            num_sequences, hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # Pool each sequence separately, respecting the attention mask
        for i in range(num_sequences):
            start_idx = start_indices[i].item()
            end_idx = end_indices[i].item() + 1  # end_indices is inclusive
            
            # Extract hidden states and attention mask for this sequence
            seq_hidden = hidden_states[start_idx:end_idx]  # (seq_len, hidden_size)
            seq_mask = attention_mask[start_idx:end_idx]  # (seq_len,)
            
            # Expand mask to match hidden dimensions
            seq_mask_expanded = seq_mask.unsqueeze(-1)  # (seq_len, 1)
            
            # Apply mask and compute mean over non-padding tokens
            # Match the reference implementation exactly (no epsilon)
            masked_hidden = seq_hidden * seq_mask_expanded
            sum_hidden = masked_hidden.sum(dim=0)  # (hidden_size,)
            sum_mask = seq_mask_expanded.sum(dim=0)  # (1,)
            
            # Compute mean - match reference implementation exactly
            pooled_outputs[i] = sum_hidden / sum_mask
        
        return pooled_outputs

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
            # Skip task-specific heads
            if name.startswith("encoder_head.") or name.startswith("scalars_") or \
               name.startswith("project_"):
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
   