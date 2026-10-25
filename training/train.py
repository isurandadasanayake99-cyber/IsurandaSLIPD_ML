"""
Phase 5: Training Loop & Model Optimization
Executes forward passes, loss calculation, backpropagation, gradient clipping,
weight updates, validation tracking, learning rate scheduling, and early stopping.
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project roots for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR / "database"))
sys.path.append(str(ROOT_DIR / "pipeline"))
sys.path.append(str(ROOT_DIR / "data"))
sys.path.append(str(ROOT_DIR / "models"))

from seed_data import generate_mock_fleet, generate_telemetry_series
from feature_pipeline import TelemetryPipeline
from dataset import split_and_scale_data, create_dataloaders
from ann_model import FleetRULPredictor, build_model_and_criterion

class FleetModelTrainer:
    """
    Coordinates training, validation, metric evaluation, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = str(ROOT_DIR / "models")
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model.to(self.device)
        self.best_val_loss = float("inf")
        self.best_model_weights = None

    def train_epoch(self, train_loader: DataLoader, max_grad_norm: float = 1.0) -> float:
        """Executes one training epoch across all mini-batches."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # 1. Zero gradients (set_to_none=True saves memory overhead)
            self.optimizer.zero_grad(set_to_none=True)

            # 2. Forward Propagation
            y_pred = self.model(X_batch)

            # 3. Loss Calculation
            loss = self.criterion(y_pred, y_batch)

            # 4. Backpropagation (Compute gradients)
            loss.backward()

            # 5. Gradient Clipping (Prevents gradient explosion on sudden sensor spikes)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

            # 6. Weight Update (AdamW step)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluates model on unseen validation data without gradient tracking.
        Calculates MSE, RMSE (in hours), and MAE.
        """
        self.model.eval()
        total_mse = 0.0
        all_preds = []
        all_targets = []

        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)

            total_mse += loss.item() * len(X_batch)
            all_preds.extend(y_pred.cpu().numpy().flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())

        num_samples = len(all_preds)
        val_mse = total_mse / max(1, num_samples)
        
        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)
        val_rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))
        val_mae = float(np.mean(np.abs(preds_arr - targets_arr)))

        return {
            "val_mse": val_mse,
            "val_rmse_hours": val_rmse,
            "val_mae_hours": val_mae
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 35,
        patience: int = 8,
        scheduler_step: bool = True
    ) -> Dict[str, list]:
        """
        Full training lifecycle with early stopping and dynamic learning rate decay.
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "lr": []
        }

        patience_counter = 0
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")

        print(f"\n[TRAIN] Beginning Training on Device: {self.device} for {epochs} Epochs...")
        print(f"{'Epoch':<8} | {'Train MSE':<12} | {'Val MSE':<12} | {'Val RMSE (h)':<14} | {'LR':<10}")
        print("-" * 65)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["val_mse"]
            val_rmse = val_metrics["val_rmse_hours"]

            current_lr = self.optimizer.param_groups[0]["lr"]

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_rmse"].append(val_rmse)
            history["lr"].append(current_lr)

            if scheduler_step:
                scheduler.step(val_loss)

            print(f"{epoch:<8} | {train_loss:<12.4f} | {val_loss:<12.4f} | {val_rmse:<14.2f} | {current_lr:<10.2e}")

            # Checkpoint Best Model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.best_model_weights,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_rmse_hours": val_rmse
                }, best_checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[EARLY STOP] Validation loss did not improve for {patience} consecutive epochs. Halting.")
                    break

        # Load best weights
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
            print(f"[SUCCESS] Loaded best weights from epoch checkpoint. Best Val MSE: {self.best_val_loss:.4f}")

        return history

def run_training_experiment():
    """Builds synthetic multi-vehicle fleet, runs pipeline, and trains ANN."""
    print("=================================================================")
    print("PHASE 5: Executing Full Deep Learning Training Pipeline")
    print("=================================================================")

    # 1. Synthesize multi-vehicle fleet telemetry
    vehicles = generate_mock_fleet(num_vehicles=5)
    all_logs = []
    for v in vehicles:
        logs = generate_telemetry_series(v["vehicle_id"], num_records=180)
        for l in logs:
            l["manufacture_year"] = v["manufacture_year"]
            l["current_odometer_km"] = v["current_odometer_km"]
            l["engine_type"] = v["engine_type"]
            l["log_id"] = 1
        all_logs.extend(logs)

    raw_df = pd.DataFrame(all_logs)

    # 2. Extract and engineer features (Phase 2)
    pipeline = TelemetryPipeline()
    processed_df, feature_cols, target_col = pipeline.run_pipeline(raw_df=raw_df, max_rul_hours=200.0)

    # 3. Scale and partition data (Phase 3)
    scaler_path = str(ROOT_DIR / "models" / "scaler.joblib")
    X_train, y_train, X_val, y_val, scaler = split_and_scale_data(
        processed_df,
        feature_cols,
        target_col,
        train_ratio=0.80,
        scaler_save_path=scaler_path
    )

    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)

    # 4. Instantiate Model & AdamW (Phase 4)
    model, criterion, optimizer = build_model_and_criterion(
        input_dim=len(feature_cols),
        lr=2e-3,
        weight_decay=1e-4
    )

    # 5. Execute Training Loop (Phase 5)
    trainer = FleetModelTrainer(model, criterion, optimizer)
    history = trainer.fit(train_loader, val_loader, epochs=25, patience=6)

    # 6. Final verification on validation set
    final_metrics = trainer.validate(val_loader)
    print("\n[EVALUATION METRICS ON UNSEEN FLEET VEHICLES]:")
    print(f"  * Validation MSE:  {final_metrics['val_mse']:.4f}")
    print(f"  * Validation RMSE: {final_metrics['val_rmse_hours']:.2f} Hours")
    print(f"  * Validation MAE:  {final_metrics['val_mae_hours']:.2f} Hours")

    return trainer, history

if __name__ == "__main__":
    run_training_experiment()
