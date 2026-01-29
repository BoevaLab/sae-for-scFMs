from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch

class ModelAdapter(ABC):
    """Abstract interface for model adapters"""
    
    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """Load the specific model"""
        pass
    
    @abstractmethod
    def preprocess_data(self, adata, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """
        Convert AnnData to model-specific PyTorch DataLoader.
        """
        pass
    
    @abstractmethod
    def setup_hook(self, modify=False) -> None:
        """Register forward hooks for activation extraction"""
        pass
    
    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run model forward pass"""
        pass
    
    @abstractmethod
    def generate_activations(self) -> Dict[str, torch.Tensor]:
        """Retrieve stored activations from hooks"""
        pass
    
    @abstractmethod
    def generate_embeddings(self, ) -> torch.Tensor:
        """Convert model-specific activations to standardized format for SAE"""
        pass

    @abstractmethod
    def clear_hooks(self, type: str = 'all') -> None:
        """Remove registered hooks"""
        pass
    
    @abstractmethod
    def prepare_compression(self, batch: Dict[str, torch.Tensor], features: torch.Tensor, 
                          n_genes: int, device: str) -> Tuple[Any, ...]:
        """Prepare model-specific data for HDF5 compression"""
        pass
    
    @abstractmethod
    def generate_activation_mask(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate mask for selecting non-padding activations from batch data"""
        pass
