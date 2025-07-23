# Unified import file for all models
# This file maintains backward compatibility while separating API and vLLM models

from ..logger import get_logger

logger = get_logger()

# Import API models directly
from .api_models import APIModel, OpenAIModel, transfer_dict_conv

# Export API models for immediate access
__all__ = [
    'transfer_dict_conv',
    'APIModel',
    'OpenAIModel',
    'CommonModelVllm',
    'InferModelVllm', 
    'Qwen3Vllm',
    'HiragVllm'
]

# Lazy loading of vLLM models
def __getattr__(name):
    """Lazy loading for vLLM models"""
    if name in ['CommonModelVllm', 'InferModelVllm', 'Qwen3Vllm', 'HiragVllm']:
        try:
            from .vllm_models import (
                CommonModelVllm,
                HiragVllm,
                InferModelVllm,
                Qwen3Vllm,
            )
            if name == 'CommonModelVllm':
                return CommonModelVllm
            elif name == 'InferModelVllm':
                return InferModelVllm
            elif name == 'Qwen3Vllm':
                return Qwen3Vllm
            elif name == 'HiragVllm':
                return HiragVllm
        except ImportError as e:
            logger.warning(f"Could not import vLLM models: {e}")
            raise AttributeError(f"vLLM models not available: {e}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 