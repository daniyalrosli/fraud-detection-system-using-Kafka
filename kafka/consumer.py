"""
Kafka Transaction Consumer with Fraud Detection.

Consumes transactions from Kafka, runs ML predictions, and stores
results in SQLite database for dashboard visualization.
"""

import os
import sys
import json
import time
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from kafka import KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.train_model import load_model_artifacts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'transactions')
CONSUMER_GROUP = os.getenv('CONSUMER_GROUP', 'fraud-detection-group')
DB_PATH = os.getenv('DB_PATH', str(Path(__file__).parent.parent / 'data' / 'predictions.db'))
MODEL_PATH = os.getenv('MODEL_PATH', str(Path(__file__).parent.parent / 'model' / 'fraud_model.pkl'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '5'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))


def init_database(db_path: str) -> sqlite3.Connection:
    """
    Initialize SQLite database for storing predictions.
    
    Args:
        db_path: Path to SQLite database file.
        
    Returns:
        Database connection.
    """
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE,
            timestamp TEXT,
            amount REAL,
            merchant_category TEXT,
            user_age INTEGER,
            account_balance REAL,
            is_fraud_actual INTEGER,
            is_fraud_predicted INTEGER,
            fraud_probability REAL,
            processed_at TEXT
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON predictions(timestamp)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_processed_at 
        ON predictions(processed_at)
    ''')
    
    conn.commit()
    logger.info(f"Database initialized at {db_path}")
    
    return conn


def create_consumer(retries: int = MAX_RETRIES) -> Optional[KafkaConsumer]:
    """
    Create a Kafka consumer with retry logic.
    
    Args:
        retries: Number of connection retries.
        
    Returns:
        KafkaConsumer instance or None if connection fails.
    """
    for attempt in range(retries):
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                group_id=CONSUMER_GROUP,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
            )
            logger.info(f"Connected to Kafka broker at {KAFKA_BROKER}")
            logger.info(f"Subscribed to topic: {KAFKA_TOPIC}")
            return consumer
        except NoBrokersAvailable:
            logger.warning(
                f"Broker not available (attempt {attempt + 1}/{retries}). "
                f"Retrying in {RETRY_DELAY}s..."
            )
            time.sleep(RETRY_DELAY)
        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
            time.sleep(RETRY_DELAY)
    
    logger.error(f"Failed to connect to Kafka after {retries} attempts")
    return None


def preprocess_transaction(
    transaction: Dict[str, Any],
    label_encoder,
    scaler
) -> np.ndarray:
    """
    Preprocess a transaction for model prediction.
    
    Args:
        transaction: Raw transaction dictionary.
        label_encoder: Fitted label encoder for merchant categories.
        scaler: Fitted StandardScaler.
        
    Returns:
        Preprocessed feature array.
    """
    # Encode merchant category
    try:
        merchant_encoded = label_encoder.transform([transaction['merchant_category']])[0]
    except ValueError:
        # Handle unknown category
        merchant_encoded = 0
        logger.warning(f"Unknown merchant category: {transaction['merchant_category']}")
    
    # Create feature array
    features = np.array([[
        transaction['amount'],
        merchant_encoded,
        transaction['user_age'],
        transaction['account_balance'],
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    return features_scaled


def predict_fraud(
    transaction: Dict[str, Any],
    model,
    label_encoder,
    scaler
) -> tuple:
    """
    Run fraud prediction on a transaction.
    
    Args:
        transaction: Transaction dictionary.
        model: Trained XGBoost model.
        label_encoder: Fitted label encoder.
        scaler: Fitted scaler.
        
    Returns:
        Tuple of (prediction, probability).
    """
    features = preprocess_transaction(transaction, label_encoder, scaler)
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return int(prediction), float(probability)


def store_prediction(
    conn: sqlite3.Connection,
    transaction: Dict[str, Any],
    prediction: int,
    probability: float
) -> bool:
    """
    Store transaction and prediction in database.
    
    Args:
        conn: Database connection.
        transaction: Transaction dictionary.
        prediction: Model prediction (0 or 1).
        probability: Fraud probability.
        
    Returns:
        True if stored successfully.
    """
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO predictions (
                transaction_id, timestamp, amount, merchant_category,
                user_age, account_balance, is_fraud_actual,
                is_fraud_predicted, fraud_probability, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            transaction['transaction_id'],
            transaction['timestamp'],
            transaction['amount'],
            transaction['merchant_category'],
            transaction['user_age'],
            transaction['account_balance'],
            transaction['is_fraud'],
            prediction,
            probability,
            datetime.now().isoformat(),
        ))
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return False


def process_messages(
    consumer: KafkaConsumer,
    conn: sqlite3.Connection,
    model_artifacts: Dict[str, Any]
) -> None:
    """
    Process incoming messages from Kafka.
    
    Args:
        consumer: Kafka consumer instance.
        conn: Database connection.
        model_artifacts: Loaded model and preprocessing objects.
    """
    model = model_artifacts['model']
    label_encoder = model_artifacts['label_encoder']
    scaler = model_artifacts['scaler']
    
    count = 0
    fraud_detected = 0
    high_risk_alerts = 0
    
    logger.info("Starting message processing loop...")
    
    try:
        for message in consumer:
            transaction = message.value
            
            # Run prediction
            prediction, probability = predict_fraud(
                transaction, model, label_encoder, scaler
            )
            
            # Store result
            store_prediction(conn, transaction, prediction, probability)
            
            count += 1
            if prediction == 1:
                fraud_detected += 1
            
            # High-risk alert
            if probability > 0.85:
                high_risk_alerts += 1
                logger.warning(
                    f"🚨 HIGH RISK ALERT - Transaction {transaction['transaction_id']}: "
                    f"${transaction['amount']:.2f} - Probability: {probability:.2%}"
                )
            
            # Log progress
            if count % 10 == 0:
                logger.info(
                    f"Processed {count} transactions | "
                    f"Fraud detected: {fraud_detected} ({fraud_detected/count:.1%}) | "
                    f"High-risk alerts: {high_risk_alerts}"
                )
                
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        logger.info(f"Consumer stopped. Total processed: {count} transactions")


def main():
    """Main entry point for the consumer."""
    logger.info("=" * 50)
    logger.info("FRAUD DETECTION - KAFKA CONSUMER")
    logger.info("=" * 50)
    logger.info(f"Broker: {KAFKA_BROKER}")
    logger.info(f"Topic: {KAFKA_TOPIC}")
    logger.info(f"Consumer Group: {CONSUMER_GROUP}")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info("=" * 50)
    
    # Load model artifacts
    try:
        model_artifacts = load_model_artifacts(MODEL_PATH)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        sys.exit(1)
    
    # Initialize database
    conn = init_database(DB_PATH)
    
    # Create consumer with retry logic
    consumer = create_consumer()
    
    if consumer is None:
        logger.error("Could not create consumer. Exiting.")
        conn.close()
        sys.exit(1)
    
    try:
        # Process messages
        process_messages(consumer, conn, model_artifacts)
    finally:
        consumer.close()
        conn.close()
        logger.info("Consumer and database connections closed.")


if __name__ == "__main__":
    main()
