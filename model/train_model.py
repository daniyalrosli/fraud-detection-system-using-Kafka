"""
Fraud Detection Model Training Module.

Trains an XGBoost classifier on synthetic transaction data with SMOTE
for handling class imbalance. Outputs evaluation metrics and saves
the trained model for production use.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.generate_transactions import generate_training_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
}


def load_or_generate_data(
    data_path: str = None,
    n_samples: int = 50000,
    fraud_rate: float = 0.05
) -> pd.DataFrame:
    """
    Load existing data or generate new synthetic dataset.
    
    Args:
        data_path: Path to existing CSV file.
        n_samples: Number of samples to generate if no file exists.
        fraud_rate: Fraud rate for synthetic data.
        
    Returns:
        DataFrame with transaction data.
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        return pd.read_csv(data_path)
    
    logger.info(f"Generating {n_samples} synthetic transactions")
    return generate_training_dataset(n_samples=n_samples, fraud_rate=fraud_rate)


def preprocess_data(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, StandardScaler]:
    """
    Preprocess data for model training.
    
    Args:
        df: Raw transaction DataFrame.
        
    Returns:
        Tuple of (features, labels, label_encoder, scaler).
    """
    logger.info("Preprocessing data...")
    
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    data['merchant_category_encoded'] = label_encoder.fit_transform(
        data['merchant_category']
    )
    
    # Select features for training
    feature_columns = [
        'amount',
        'merchant_category_encoded',
        'user_age',
        'account_balance',
    ]
    
    X = data[feature_columns].values
    y = data['is_fraud'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Features shape: {X_scaled.shape}")
    logger.info(f"Fraud rate: {y.mean():.2%}")
    
    return X_scaled, y, label_encoder, scaler


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to handle class imbalance.
    
    Args:
        X: Feature matrix.
        y: Labels.
        random_state: Random seed for reproducibility.
        
    Returns:
        Resampled (X, y) tuple.
    """
    logger.info("Applying SMOTE for class imbalance...")
    logger.info(f"Before SMOTE - Class distribution: {np.bincount(y)}")
    
    smote = SMOTE(random_state=random_state, sampling_strategy=0.5)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    logger.info(f"After SMOTE - Class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any] = None
) -> XGBClassifier:
    """
    Train XGBoost classifier.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        config: Model configuration dictionary.
        
    Returns:
        Trained XGBClassifier.
    """
    if config is None:
        config = MODEL_CONFIG
    
    logger.info("Training XGBoost model...")
    
    model = XGBClassifier(**config)
    model.fit(X_train, y_train)
    
    logger.info("Model training complete")
    
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("Evaluating model performance...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
    }
    
    # Print detailed report
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    print("\nKey Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("=" * 50 + "\n")
    
    return metrics


def save_model_artifacts(
    model: XGBClassifier,
    label_encoder: LabelEncoder,
    scaler: StandardScaler,
    output_dir: str = None
) -> str:
    """
    Save model and preprocessing artifacts.
    
    Args:
        model: Trained model.
        label_encoder: Fitted label encoder.
        scaler: Fitted scaler.
        output_dir: Directory to save artifacts.
        
    Returns:
        Path to saved model file.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all artifacts in a single file
    artifacts = {
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_names': ['amount', 'merchant_category_encoded', 'user_age', 'account_balance'],
    }
    
    model_path = output_dir / 'fraud_model.pkl'
    joblib.dump(artifacts, model_path)
    logger.info(f"Model artifacts saved to {model_path}")
    
    return str(model_path)


def load_model_artifacts(model_path: str) -> Dict[str, Any]:
    """
    Load model and preprocessing artifacts.
    
    Args:
        model_path: Path to saved model file.
        
    Returns:
        Dictionary containing model and preprocessing objects.
    """
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


def main():
    """Main training pipeline."""
    logger.info("Starting fraud detection model training pipeline")
    
    # Load or generate data
    df = load_or_generate_data(n_samples=50000, fraud_rate=0.05)
    
    # Preprocess
    X, y, label_encoder, scaler = preprocess_data(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to training data only
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    # Train model
    model = train_model(X_train_balanced, y_train_balanced)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Cross-validation
    logger.info("Running cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model
    model_path = save_model_artifacts(model, label_encoder, scaler)
    
    logger.info("Training pipeline complete!")
    
    return model, metrics


if __name__ == "__main__":
    main()
