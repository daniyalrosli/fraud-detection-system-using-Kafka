"""
Standalone Pipeline Runner (No Kafka Required).

This script runs the entire fraud detection pipeline without Kafka,
using a simple queue-based approach for testing and demonstration.
"""

import os
import sys
import json
import time
import sqlite3
import logging
import threading
from queue import Queue
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.generate_transactions import generate_single_transaction
from model.train_model import load_model_artifacts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = os.getenv('DB_PATH', str(Path(__file__).parent / 'data' / 'predictions.db'))
MODEL_PATH = os.getenv('MODEL_PATH', str(Path(__file__).parent / 'model' / 'fraud_model.pkl'))
TRANSACTION_INTERVAL = float(os.getenv('TRANSACTION_INTERVAL', '1.0'))

# Shared queue
transaction_queue = Queue()


def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
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
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_at ON predictions(processed_at)')
    conn.commit()
    
    return conn


def producer_thread(stop_event: threading.Event):
    """Generate and queue transactions."""
    logger.info("Producer started - generating transactions...")
    count = 0
    
    while not stop_event.is_set():
        transaction = generate_single_transaction(
            base_time=datetime.now(),
            fraud_rate=0.05
        )
        transaction_queue.put(transaction)
        count += 1
        
        if count % 10 == 0:
            logger.info(f"Producer: Generated {count} transactions")
        
        time.sleep(TRANSACTION_INTERVAL)
    
    logger.info(f"Producer stopped. Total generated: {count}")


def consumer_thread(stop_event: threading.Event, model_artifacts: dict, conn: sqlite3.Connection):
    """Process transactions and run predictions."""
    logger.info("Consumer started - processing transactions...")
    
    model = model_artifacts['model']
    label_encoder = model_artifacts['label_encoder']
    scaler = model_artifacts['scaler']
    
    count = 0
    fraud_detected = 0
    
    while not stop_event.is_set():
        try:
            # Get transaction with timeout
            transaction = transaction_queue.get(timeout=1.0)
            
            # Preprocess
            try:
                merchant_encoded = label_encoder.transform([transaction['merchant_category']])[0]
            except ValueError:
                merchant_encoded = 0
            
            features = np.array([[
                transaction['amount'],
                merchant_encoded,
                transaction['user_age'],
                transaction['account_balance'],
            ]])
            features_scaled = scaler.transform(features)
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            # Store in database
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
                int(prediction),
                float(probability),
                datetime.now().isoformat(),
            ))
            conn.commit()
            
            count += 1
            if prediction == 1:
                fraud_detected += 1
            
            # Alert on high probability
            if probability > 0.85:
                logger.warning(
                    f"🚨 HIGH RISK: {transaction['transaction_id']} | "
                    f"${transaction['amount']:.2f} | Prob: {probability:.1%}"
                )
            
            if count % 10 == 0:
                logger.info(
                    f"Consumer: Processed {count} | "
                    f"Fraud: {fraud_detected} ({fraud_detected/count:.1%})"
                )
                
        except Exception:
            continue
    
    logger.info(f"Consumer stopped. Total processed: {count}")


def main():
    """Run the standalone pipeline."""
    print("\n" + "=" * 60)
    print("🔒 REAL-TIME FRAUD DETECTION PIPELINE (Standalone Mode)")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Interval: {TRANSACTION_INTERVAL}s")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the pipeline\n")
    
    # Load model
    try:
        model_artifacts = load_model_artifacts(MODEL_PATH)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error(f"Model not found. Run: python model/train_model.py")
        sys.exit(1)
    
    # Initialize database
    conn = init_database(DB_PATH)
    logger.info("Database initialized")
    
    # Create stop event
    stop_event = threading.Event()
    
    # Start threads
    producer = threading.Thread(target=producer_thread, args=(stop_event,))
    consumer = threading.Thread(target=consumer_thread, args=(stop_event, model_artifacts, conn))
    
    producer.start()
    consumer.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("\nShutting down pipeline...")
        stop_event.set()
        producer.join()
        consumer.join()
        conn.close()
        logger.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
