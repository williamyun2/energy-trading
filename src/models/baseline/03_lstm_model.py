"""
LSTM Model with Attention - ERCOT DAM Price Forecasting
ML Model #3: Deep Learning Time Series Model

Optimized for NVIDIA H100 GPUs with:
- Mixed precision training (FP16)
- Attention mechanism
- Early stopping
- TensorBoard logging

Goal: Beat Linear Regression (MAE < $9.27/MWh)
Target: MAE $5-6/MWh, R² > 0.80
"""

import pandas as pd
import numpy as np


import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json



from pathlib import Path
import sys

# Robust: works no matter where you run python from
project_root = Path(__file__).resolve().parents[3]  # .../power_trading
src_dir = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from merge_dataset.loader import load_clean_data

# Import feature engineering
from models.feature_engineering import engineer_all_features

# Output directory
RESULTS_DIR = Path(r"D:\Users\williamyun\proj\power_trading\results\lstm")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"🎮 GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Available GPUs: {torch.cuda.device_count()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  No GPU detected, using CPU (training will be slow)")


class TimeSeriesDataset(Dataset):
    """Dataset for time series with sliding window"""
    
    def __init__(self, features, targets, sequence_length=168):
        """
        Args:
            features: numpy array of shape (n_samples, n_features)
            targets: numpy array of shape (n_samples,)
            sequence_length: number of time steps to look back (default: 168 = 1 week)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        X = self.features[idx:idx + self.sequence_length]
        # Get target (next time step)
        y = self.targets[idx + self.sequence_length]
        return X, y


class AttentionLayer(nn.Module):
    """Attention mechanism to focus on important time steps"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch, seq_len, 1)
        
        # Apply attention weights
        context = torch.sum(attention_weights * lstm_output, dim=1)
        # context shape: (batch, hidden_size)
        
        return context, attention_weights


class LSTMPricePredictor(nn.Module):
    """LSTM with Attention for price prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super(LSTMPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size)
        
        # Attention
        context, attention_weights = self.attention(lstm_out)
        # context shape: (batch, hidden_size)
        
        # Fully connected layers
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out.squeeze()


def select_features(df):
    """Select features for modeling"""
    
    exclude_cols = ['datetime', 'dam_price']
    
    # Exclude original raw weather columns (use derived features)
    raw_weather = [c for c in df.columns if any(x in c for x in [
        'temp_f_HOUSTON', 'temp_f_NORTH', 'temp_f_SOUTH', 'temp_f_WEST',
        'wind_speed_mph_HOUSTON', 'wind_speed_mph_NORTH', 'wind_speed_mph_SOUTH', 'wind_speed_mph_WEST',
        'solar_radiation_wm2_HOUSTON', 'solar_radiation_wm2_NORTH', 'solar_radiation_wm2_SOUTH', 'solar_radiation_wm2_WEST',
        'relative_humidity_HOUSTON', 'relative_humidity_NORTH', 'relative_humidity_SOUTH', 'relative_humidity_WEST'
    ])]
    
    exclude_cols.extend(raw_weather)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    return feature_cols


def train_test_split(df, test_months=6):
    """Split data into train/test sets (temporal split)"""
    
    max_date = df['datetime'].max()
    split_date = max_date - pd.DateOffset(months=test_months)
    
    df_train = df[df['datetime'] < split_date].copy()
    df_test = df[df['datetime'] >= split_date].copy()
    
    return df_train, df_test, split_date


def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch with mixed precision"""
    
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
                patience=10, device=device):
    """Train model with early stopping"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # For mixed precision
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    print(f"\n{'='*80}")
    print("TRAINING LSTM MODEL")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Mixed Precision: Enabled (FP16)")
    print(f"Max Epochs: {num_epochs}")
    print(f"Early Stopping Patience: {patience}")
    print(f"Learning Rate: {learning_rate}")
    print()
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Record history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save(model.state_dict(), RESULTS_DIR / 'best_model.pt')
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(RESULTS_DIR / 'best_model.pt'))
    
    return model, training_history


def predict(model, data_loader, device):
    """Generate predictions"""
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            batch_pred = model(X_batch)
            predictions.extend(batch_pred.cpu().numpy())
    
    return np.array(predictions)


def evaluate_model(y_true, y_pred, model_name="LSTM"):
    """Calculate evaluation metrics"""
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    max_error = np.max(np.abs(y_true - y_pred))
    median_error = np.median(np.abs(y_true - y_pred))
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Max Error': max_error,
        'Median Error': median_error
    }


def plot_training_history(history, results_dir):
    """Plot training and validation loss"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training History', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].plot(epochs, history['learning_rate'], linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions(df_test, y_pred, results_dir, sequence_length):
    """Create diagnostic plots"""
    
    # Adjust for sequence length
    df_plot = df_test.iloc[sequence_length:].copy()
    df_plot = df_plot.iloc[:len(y_pred)].copy()
    df_plot['predicted'] = y_pred
    df_plot['error'] = df_plot['dam_price'] - df_plot['predicted']
    df_plot['abs_error'] = np.abs(df_plot['error'])
    
    sns.set_style("whitegrid")
    
    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df_plot['dam_price'], df_plot['predicted'], alpha=0.3, s=10)
    
    min_val = min(df_plot['dam_price'].min(), df_plot['predicted'].min())
    max_val = max(df_plot['dam_price'].max(), df_plot['predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Price ($/MWh)', fontsize=12)
    ax.set_ylabel('Predicted Price ($/MWh)', fontsize=12)
    ax.set_title('LSTM: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / '01_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Error Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].hist(df_plot['error'], bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Prediction Error ($/MWh)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df_plot['datetime'], df_plot['error'], alpha=0.5, linewidth=0.5)
    axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Prediction Error ($/MWh)')
    axes[0, 1].set_title('Errors Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    axes[1, 0].plot(df_plot['datetime'], df_plot['abs_error'], alpha=0.5, linewidth=0.5, color='orange')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Absolute Error ($/MWh)')
    axes[1, 0].set_title('Absolute Errors Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].scatter(df_plot['dam_price'], df_plot['abs_error'], alpha=0.3, s=10)
    axes[1, 1].set_xlabel('Actual Price ($/MWh)')
    axes[1, 1].set_ylabel('Absolute Error ($/MWh)')
    axes[1, 1].set_title('Error vs Actual Price')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / '02_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Last 30 days
    df_recent = df_plot.tail(30 * 24)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df_recent['datetime'], df_recent['dam_price'], label='Actual', linewidth=2, alpha=0.8)
    ax.plot(df_recent['datetime'], df_recent['predicted'], label='Predicted', linewidth=2, alpha=0.8)
    ax.fill_between(df_recent['datetime'], df_recent['dam_price'], df_recent['predicted'], 
                     alpha=0.3, label='Error')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($/MWh)', fontsize=12)
    ax.set_title('Last 30 Days: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir / '03_last_30_days.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main training and evaluation pipeline"""
    
    print("="*80)
    print("LSTM MODEL - ERCOT DAM PRICE FORECASTING")
    print("="*80)
    print("Goal: Beat Linear Regression (MAE < $9.27/MWh)")
    print("Target: MAE $5-6/MWh, R² > 0.80")
    print()
    
    # Hyperparameters
    SEQUENCE_LENGTH = 168  # 1 week lookback
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    DROPOUT = 0.2
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 10
    
    # 1. Load data
    print("Loading clean data...")
    df = load_clean_data(verbose=False)
    print(f"✓ Loaded {len(df):,} records")
    
    # 2. Engineer features
    print("\nEngineering features...")
    df_features = engineer_all_features(df, verbose=False)
    print(f"✓ Created {df_features.shape[1] - 20} new features")
    
    # 3. Select features
    feature_cols = select_features(df_features)
    print(f"✓ Selected {len(feature_cols)} features")
    
    # 4. Handle missing values
    print("\nHandling missing values...")
    df_features = df_features.dropna()
    print(f"✓ {len(df_features):,} rows after dropping NAs")
    
    # 5. Train/test split
    print("\nSplitting train/test...")
    df_train, df_test, split_date = train_test_split(df_features, test_months=6)
    print(f"✓ Train: {len(df_train):,} records")
    print(f"✓ Test:  {len(df_test):,} records")
    
    # 6. Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols])
    X_test = scaler.transform(df_test[feature_cols])
    y_train = df_train['dam_price'].values
    y_test = df_test['dam_price'].values
    print(f"✓ Features scaled")
    
    # 7. Create datasets
    print(f"\nCreating time series datasets (sequence length: {SEQUENCE_LENGTH})...")
    train_dataset = TimeSeriesDataset(X_train, y_train, SEQUENCE_LENGTH)
    test_dataset = TimeSeriesDataset(X_test, y_test, SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✓ Train sequences: {len(train_dataset):,}")
    print(f"✓ Test sequences: {len(test_dataset):,}")
    
    # 8. Create model
    print(f"\nCreating LSTM model...")
    model = LSTMPricePredictor(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # 9. Train model
    model, history = train_model(
        model, train_loader, test_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        device=device
    )
    
    # 10. Generate predictions
    print("\nGenerating predictions on test set...")
    y_pred = predict(model, test_loader, device)
    
    # Align predictions with actual test data
    y_test_aligned = y_test[SEQUENCE_LENGTH:SEQUENCE_LENGTH + len(y_pred)]
    
    print(f"✓ Generated {len(y_pred):,} predictions")
    
    # 11. Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(y_test_aligned, y_pred, model_name='LSTM')
    
    # Compare to baselines
    baseline_mae = 10.97
    linear_mae = 9.27
    improvement_baseline = (baseline_mae - metrics['MAE']) / baseline_mae * 100
    improvement_linear = (linear_mae - metrics['MAE']) / linear_mae * 100
    
    print("\n" + "="*80)
    print("LSTM RESULTS")
    print("="*80)
    print(f"Model: {metrics['Model']}")
    print(f"MAE:          ${metrics['MAE']:.2f}/MWh")
    print(f"RMSE:         ${metrics['RMSE']:.2f}/MWh")
    print(f"MAPE:         {metrics['MAPE']:.2f}%")
    print(f"R² Score:     {metrics['R2']:.4f}")
    print(f"Max Error:    ${metrics['Max Error']:.2f}/MWh")
    print(f"Median Error: ${metrics['Median Error']:.2f}/MWh")
    print()
    print("="*80)
    print("COMPARISON TO BASELINES")
    print("="*80)
    print(f"Persistence Baseline MAE:    ${baseline_mae:.2f}/MWh")
    print(f"Linear Regression MAE:       ${linear_mae:.2f}/MWh")
    print(f"LSTM MAE:                    ${metrics['MAE']:.2f}/MWh")
    print()
    print(f"Improvement over Persistence: {improvement_baseline:+.1f}%")
    print(f"Improvement over Linear Reg:  {improvement_linear:+.1f}%")
    
    if metrics['MAE'] < linear_mae:
        print(f"\n✅ SUCCESS! Beat Linear Regression by ${linear_mae - metrics['MAE']:.2f}/MWh")
    else:
        print(f"\n❌ Did not beat Linear Regression (worse by ${metrics['MAE'] - linear_mae:.2f}/MWh)")
    
    # 12. Create plots
    print("\nCreating diagnostic plots...")
    plot_training_history(history, RESULTS_DIR)
    plot_predictions(df_test, y_pred, RESULTS_DIR, SEQUENCE_LENGTH)
    print(f"✓ Saved plots to: {RESULTS_DIR}")
    
    # 13. Save results
    print("\nSaving results...")
    
    # Metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(RESULTS_DIR / "lstm_metrics.csv", index=False)
    
    # Predictions
    df_pred = df_test.iloc[SEQUENCE_LENGTH:SEQUENCE_LENGTH + len(y_pred)].copy()
    df_pred = df_pred[['datetime', 'dam_price']].copy()
    df_pred['predicted'] = y_pred
    df_pred['error'] = df_pred['dam_price'] - df_pred['predicted']
    df_pred.to_csv(RESULTS_DIR / "lstm_predictions.csv", index=False)
    
    # Training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)
    
    # Model config
    config = {
        'sequence_length': SEQUENCE_LENGTH,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'total_params': total_params,
        'device': str(device),
        'mixed_precision': True
    }
    with open(RESULTS_DIR / 'model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Saved metrics to: {RESULTS_DIR / 'lstm_metrics.csv'}")
    print(f"✓ Saved predictions to: {RESULTS_DIR / 'lstm_predictions.csv'}")
    print(f"✓ Saved training history to: {RESULTS_DIR / 'training_history.csv'}")
    print(f"✓ Saved model config to: {RESULTS_DIR / 'model_config.json'}")
    print(f"✓ Saved best model to: {RESULTS_DIR / 'best_model.pt'}")
    
    print("\n" + "="*80)
    print("LSTM MODEL COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - lstm_metrics.csv")
    print(f"  - lstm_predictions.csv")
    print(f"  - training_history.csv")
    print(f"  - model_config.json")
    print(f"  - best_model.pt")
    print(f"  - training_history.png")
    print(f"  - 01_actual_vs_predicted.png")
    print(f"  - 02_error_analysis.png")
    print(f"  - 03_last_30_days.png")
    
    return metrics, model


if __name__ == "__main__":
    metrics, model = main()