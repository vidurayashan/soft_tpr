import torch 
import numpy as np

class RSquared:
    def __init__(self, normalized_labels: np.ndarray, device: str = 'cpu'):
        variance_per_factor = ((normalized_labels - normalized_labels.mean(
            axis=0, keepdims=True)) ** 2).mean(axis=0)
        self.variance_per_factor = torch.tensor(variance_per_factor).to(device)

    def __call__(self, predictions: torch.tensor,
                 targets: torch.tensor) -> torch.tensor:
        assert predictions.shape == targets.shape
        assert len(targets.shape) == 2
        mse_loss_per_factor = (predictions - targets).pow(2).mean(dim=0)
        return 1 - mse_loss_per_factor / self.variance_per_factor