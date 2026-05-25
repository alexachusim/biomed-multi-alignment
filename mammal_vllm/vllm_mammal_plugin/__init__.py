"""vLLM MAMMAL model plugin.

This plugin registers the MAMMAL model with vLLM's ModelRegistry,
allowing it to be used with vLLM's inference engine.
"""

from vllm.logger import init_logger
from transformers import PretrainedConfig
from transformers import AutoConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm_mammal_plugin.mammal import MammalConfig

__version__ = "0.1.0"


def register_mammal_model() -> None:
    """Register MAMMAL models with vLLM's ModelRegistry.

    This function is called automatically when the plugin is loaded
    through vLLM's plugin discovery mechanism.
    """
    logger = init_logger(__name__)
    try:               
        AutoConfig.register("t5", MammalConfig, exist_ok=True)

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

    # Intercept Transformers configuration loading
    original_get_config_dict = PretrainedConfig.get_config_dict.__func__

    def patched_get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        # Call the original loader to pull down the config.json dictionary
        config_dict, kwargs_out = original_get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
        
        # Check if model_type is missing or if this is your specific model repo
        if "model_type" not in config_dict:
            logger.info("Patching missing 'model_type' into the loaded configuration dictionary")
            config_dict["model_type"] = "t5"          
                
        return config_dict, kwargs_out

    # Overwrite the class method globally across the running process
    PretrainedConfig.get_config_dict = classmethod(patched_get_config_dict)  # type: ignore[method-assign]


__all__ = [
    "register_mammal_model",
    "__version__",
]
