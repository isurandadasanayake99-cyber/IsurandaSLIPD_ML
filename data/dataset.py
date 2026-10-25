"""
Phase 3: Data Preparation Module
Constructs the custom PyTorch Dataset, handles leak-free vehicle splits,
applies z-score feature scaling, persists scalers, and generates DataLoaders.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader

class TelemetryDataset(Dataset):
    """
    Custom PyTorch Dataset for fleet sensor telemetry.
    Yields (X, y) tensor pairs formatted as torch.float32.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Convert to 32-bit floats for PyTorch compatibility
        self.X = torch.tensor(X, dtype=torch.float32)
        # Target shape is (N, 1) for regression
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

def split_and_scale_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_rul_hours",
    train_ratio: float = 0.75,
    scaler_save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Splits fleet data without temporal leakage using vehicle group partitioning.
    Fits StandardScaler on train set only and transforms both train and validation sets.
    """
    unique_vehicles = df["vehicle_id"].unique()
    num_train = max(1, int(len(unique_vehicles) * train_ratio))
    
    train_vehicles = unique_vehicles[:num_train]
    val_vehicles = unique_vehicles[num_train:]
    
    # If only 1 vehicle in dataset, fallback to causal chronological time-split
    if len(val_vehicles) == 0:
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        print(f"[SPLIT] Single vehicle detected: Chronological time split applied ({train_ratio:.0%}/{1-train_ratio:.0%}).")
    else:
        train_df = df[df["vehicle_id"].isin(train_vehicles)].copy()
        val_df = df[df["vehicle_id"].isin(val_vehicles)].copy()
        print(f"[SPLIT] Group Disjoint Split -> Train Vehicles: {list(train_vehicles)}, Val Vehicles: {list(val_vehicles)}")

    X_train_raw = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_val_raw = val_df[feature_cols].values
    y_val = val_df[target_col].values

    # Fit scaler ONLY on training data to prevent distribution snooping
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    # Persist the fitted scaler for Phase 6 production inference
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"[SCALER] Saved fitted StandardScaler to: {scaler_save_path}")

    return X_train_scaled, y_train, X_val_scaled, y_val, scaler

def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Wraps NumPy arrays into TelemetryDatasets and builds PyTorch DataLoaders.
    """
    train_dataset = TelemetryDataset(X_train, y_train)
    val_dataset = TelemetryDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Shuffle train mini-batches to break correlation between consecutive batches
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # Sequential validation evaluation
        drop_last=False
    )

    return train_loader, val_loader

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    root_dir = Path(__file__).parent.parent
    sys.path.append(str(root_dir / "database"))
    sys.path.append(str(root_dir / "pipeline"))

    from seed_data import generate_mock_fleet, generate_telemetry_series
    from feature_pipeline import TelemetryPipeline

    print("Running Phase 3 Verification Test...")
    # 1. Generate multi-vehicle data
    vehicles = generate_mock_fleet(num_vehicles=4)
    raw_logs = []
    for v in vehicles:
        logs = generate_telemetry_series(v["vehicle_id"], num_records=120)
        for l in logs:
            l["manufacture_year"] = v["manufacture_year"]
            l["current_odometer_km"] = v["current_odometer_km"]
            l["engine_type"] = v["engine_type"]
            l["log_id"] = 1
        raw_logs.extend(logs)

    # 2. Extract features via Phase 2 pipeline
    pipeline = TelemetryPipeline()
    processed_df, feature_cols, target_col = pipeline.run_pipeline(raw_df=pd.DataFrame(raw_logs))

    # 3. Split, Scale & Build PyTorch DataLoaders
    scaler_path = str(root_dir / "models" / "scaler.joblib")
    X_train, y_train, X_val, y_val, scaler = split_and_scale_data(
        processed_df,
        feature_cols,
        target_col,
        train_ratio=0.75,
        scaler_save_path=scaler_path
    )

    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)

    print(f"\n[SUMMARY] Train Samples: {len(X_train)}, Val Samples: {len(X_val)}")
    print(f"[SUMMARY] Number of Features: {len(feature_cols)}")
    print(f"[SUMMARY] Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}")

    # 4. Inspect single mini-batch
    sample_X, sample_y = next(iter(train_loader))
    print(f"\n[BATCH TEST] X shape: {sample_X.shape} (dtype: {sample_X.dtype})")
    print(f"[BATCH TEST] y shape: {sample_y.shape} (dtype: {sample_y.dtype})")
    print(f"[BATCH TEST] X sample mean ~0: {sample_X.mean():.4f}, std ~1: {sample_X.std():.4f}")
