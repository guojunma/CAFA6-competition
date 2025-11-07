"""Machine learning models for GO term prediction."""

from .baseline_ml import BaselineMLModel

__all__ = ['BaselineMLModel']

# Optional deep learning imports (requires torch)
try:
    from .deep_learning import DeepLearningModel, ProteinCNN
    __all__.extend(['DeepLearningModel', 'ProteinCNN'])
except ImportError:
    pass  # torch not installed, deep learning models not available
