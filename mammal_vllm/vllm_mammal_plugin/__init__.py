"""vLLM MAMMAL model plugin.

This plugin registers the MAMMAL model with vLLM's ModelRegistry,
allowing it to be used with vLLM's inference engine.
"""

from vllm.logger import init_logger
from vllm.model_executor.models.registry import ModelRegistry

__version__ = "0.1.0"


def register_mammal_model() -> None:
    """Register MAMMAL models with vLLM's ModelRegistry.

    This function is called automatically when the plugin is loaded
    through vLLM's plugin discovery mechanism.
    """
    logger = init_logger(__name__)
    try:        
        # Register T5ForConditionalGeneration with the ModelRegistry
        # Using lazy loading to avoid importing the model class during plugin discovery
        ModelRegistry.register_model(
            "T5ForConditionalGeneration",
            "vllm_mammal_plugin.mammal:T5ForConditionalGeneration",
        )  
        logger.info("Successfully registered MAMMAL model with vLLM")

    except Exception as e:
        logger.error(f"Failed to register MAMMAL model: {e}")
        raise


__all__ = [
    "register_mammal_model",
    "__version__",
]
