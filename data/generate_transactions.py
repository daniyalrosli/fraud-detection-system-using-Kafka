"""
Synthetic Transaction Data Generator for Fraud Detection System.

This module generates realistic credit card transaction data with configurable
fraud rates and realistic patterns for training ML models and testing pipelines.
"""

import random
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Generator

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Merchant categories with associated fraud risk weights
MERCHANT_CATEGORIES = {
    'grocery': 0.02,
    'gas_station': 0.03,
    'restaurant': 0.02,
    'online_shopping': 0.08,
    'travel': 0.06,
    'entertainment': 0.03,
    'electronics': 0.10,
    'jewelry': 0.12,
    'atm_withdrawal': 0.05,
    'money_transfer': 0.15,
}


def generate_transaction_id() -> str:
    """Generate a unique transaction ID."""
    return str(uuid.uuid4())[:12].upper()


def generate_amount(is_fraud: bool) -> float:
    """
    Generate a transaction amount based on fraud status.
    
    Fraudulent transactions tend to be higher amounts or very small test amounts.
    
    Args:
        is_fraud: Whether this transaction is fraudulent.
        
    Returns:
        Transaction amount rounded to 2 decimal places.
    """
    if is_fraud:
        # Fraudulent: either very small (testing) or large amounts
        if random.random() < 0.3:
            amount = random.uniform(0.01, 5.00)  # Test transactions
        else:
            amount = random.uniform(500, 10000)  # Large fraudulent amounts
    else:
        # Legitimate: follows log-normal distribution (realistic spending)
        amount = np.random.lognormal(mean=3.5, sigma=1.2)
        amount = min(amount, 5000)  # Cap at reasonable max
    
    return round(amount, 2)


def generate_user_age() -> int:
    """Generate a realistic user age (18-85)."""
    return int(np.random.normal(loc=42, scale=15))


def generate_account_balance(is_fraud: bool) -> float:
    """
    Generate account balance.
    
    Fraudulent transactions often occur on accounts with lower balances
    or newly created accounts.
    
    Args:
        is_fraud: Whether this transaction is fraudulent.
        
    Returns:
        Account balance rounded to 2 decimal places.
    """
    if is_fraud and random.random() < 0.4:
        # Some fraud on low-balance accounts
        balance = random.uniform(10, 500)
    else:
        balance = np.random.lognormal(mean=8, sigma=1.5)
        balance = min(balance, 100000)
    
    return round(balance, 2)


def generate_timestamp(base_time: Optional[datetime] = None) -> str:
    """
    Generate an ISO format timestamp.
    
    Args:
        base_time: Base time to use. If None, uses current time.
        
    Returns:
        ISO format timestamp string.
    """
    if base_time is None:
        base_time = datetime.now()
    return base_time.isoformat()


def determine_fraud(merchant_category: str, base_fraud_rate: float = 0.05) -> bool:
    """
    Determine if a transaction is fraudulent based on merchant risk.
    
    Args:
        merchant_category: The merchant category of the transaction.
        base_fraud_rate: Base fraud rate (default 5%).
        
    Returns:
        True if transaction is fraudulent, False otherwise.
    """
    category_risk = MERCHANT_CATEGORIES.get(merchant_category, 0.05)
    adjusted_rate = base_fraud_rate * (1 + category_risk * 5)
    return random.random() < adjusted_rate


def generate_single_transaction(
    base_time: Optional[datetime] = None,
    fraud_rate: float = 0.05
) -> Dict[str, Any]:
    """
    Generate a single synthetic transaction.
    
    Args:
        base_time: Base timestamp for the transaction.
        fraud_rate: Target fraud rate.
        
    Returns:
        Dictionary containing transaction data.
    """
    merchant_category = random.choice(list(MERCHANT_CATEGORIES.keys()))
    is_fraud = determine_fraud(merchant_category, fraud_rate)
    
    user_age = generate_user_age()
    user_age = max(18, min(85, user_age))  # Clamp to valid range
    
    transaction = {
        'transaction_id': generate_transaction_id(),
        'timestamp': generate_timestamp(base_time),
        'amount': generate_amount(is_fraud),
        'merchant_category': merchant_category,
        'user_age': user_age,
        'account_balance': generate_account_balance(is_fraud),
        'is_fraud': int(is_fraud),
    }
    
    return transaction


def generate_transactions_batch(
    n_transactions: int,
    fraud_rate: float = 0.05,
    time_span_hours: int = 24
) -> Generator[Dict[str, Any], None, None]:
    """
    Generate a batch of synthetic transactions.
    
    Args:
        n_transactions: Number of transactions to generate.
        fraud_rate: Target fraud rate.
        time_span_hours: Time span over which to distribute transactions.
        
    Yields:
        Transaction dictionaries.
    """
    logger.info(f"Generating {n_transactions} transactions with {fraud_rate:.1%} fraud rate")
    
    start_time = datetime.now() - timedelta(hours=time_span_hours)
    time_increment = timedelta(hours=time_span_hours) / n_transactions
    
    for i in range(n_transactions):
        transaction_time = start_time + (time_increment * i)
        yield generate_single_transaction(transaction_time, fraud_rate)
    
    logger.info("Batch generation complete")


def generate_training_dataset(
    n_samples: int = 10000,
    fraud_rate: float = 0.05,
    output_path: Optional[str] = None
) -> "pd.DataFrame":
    """
    Generate a training dataset for the ML model.
    
    Args:
        n_samples: Number of samples to generate.
        fraud_rate: Target fraud rate.
        output_path: Optional path to save CSV file.
        
    Returns:
        Pandas DataFrame with transaction data.
    """
    import pandas as pd
    
    logger.info(f"Generating training dataset with {n_samples} samples")
    
    transactions = list(generate_transactions_batch(n_samples, fraud_rate))
    df = pd.DataFrame(transactions)
    
    fraud_count = df['is_fraud'].sum()
    logger.info(f"Generated {fraud_count} fraudulent transactions ({fraud_count/n_samples:.1%})")
    
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Demo: Generate sample transactions
    print("\n=== Sample Transactions ===\n")
    
    for i, txn in enumerate(generate_transactions_batch(5, fraud_rate=0.05)):
        fraud_label = "🚨 FRAUD" if txn['is_fraud'] else "✓ Legit"
        print(f"Transaction {i+1}: {fraud_label}")
        for key, value in txn.items():
            print(f"  {key}: {value}")
        print()
    
    # Generate training dataset
    print("\n=== Generating Training Dataset ===\n")
    df = generate_training_dataset(n_samples=1000, output_path="sample_transactions.csv")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFraud distribution:\n{df['is_fraud'].value_counts()}")
