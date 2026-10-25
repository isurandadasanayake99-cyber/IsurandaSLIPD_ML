"""
Phase 4: Deep Learning Model Architecture
Implements the FleetRULPredictor Artificial Neural Network (ANN/MLP)
with Batch Normalization, Dropout regularization, and Kaiming He weight initialization.
"""

import torch
import torch.nn as nn
from typing import List, Optional

class FleetRULPredictor(nn.Module):
    """
    Multi-Layer Perceptron (ANN) for continuous Remaining Useful Life (RUL) regression.
    
    Architecture:
        Input Layer (D_in)
        -> [Linear -> BatchNorm1d -> LeakyReLU -> Dropout] x N
        -> Output Layer (Linear -> 1)
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        negative_slope: float = 0.01
    ):
        super(FleetRULPredictor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        # Build hidden layer blocks
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim

        # Final regression head (unconstrained scalar output)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        
        # Apply Kaiming (He) normal initialization suited for LeakyReLU activations
        self._initialize_weights(negative_slope)

    def _initialize_weights(self, negative_slope: float):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=negative_slope, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            x: Tensor of shape (batch_size, input_dim)
            Returns: Tensor of shape (batch_size, 1) representing predicted RUL (hours)
        """
        return self.network(x)

    def get_parameter_count(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model_and_criterion(
    input_dim: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_huber: bool = False,
    huber_delta: float = 10.0
):
    """
    Factory function instantiating the model, criterion, and AdamW optimizer.
    """
    model = FleetRULPredictor(input_dim=input_dim)
    
    if use_huber:
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    return model, criterion, optimizer

if __name__ == "__main__":
    print("Testing FleetRULPredictor Architecture...")
    input_features = 30
    batch_size = 32

    model, criterion, optimizer = build_model_and_criterion(input_dim=input_features)
    
    print(model)
    print(f"\n[INFO] Total Trainable Parameters: {model.get_parameter_count():,}")

    # Forward pass smoke test with synthetic batch
    mock_input = torch.randn(batch_size, input_features, dtype=torch.float32)
    mock_target = torch.rand(batch_size, 1, dtype=torch.float32) * 200.0

    # Ensure evaluation mode disables Dropout / BatchNorm running stats tracking
    model.eval()
    with torch.no_grad():
        preds = model(mock_input)
    print(f"[TEST] Forward pass successful. Output Shape: {preds.shape} (Expected: ({batch_size}, 1))")

    # Training mode test
    model.train()
    optimizer.zero_grad()
    train_preds = model(mock_input)
    loss = criterion(train_preds, mock_target)
    loss.backward()
    optimizer.step()
    print(f"[TEST] Backward pass & AdamW step executed. Initial Batch Loss: {loss.item():.4f}")
