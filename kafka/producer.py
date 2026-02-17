"""
Kafka Transaction Producer.

Streams synthetic transaction data to a Kafka topic for real-time
fraud detection processing. Includes retry logic and error handling.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.generate_transactions import generate_single_transaction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'transactions')
TRANSACTION_INTERVAL = float(os.getenv('TRANSACTION_INTERVAL', '1.0'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '5'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))


def create_producer(retries: int = MAX_RETRIES) -> Optional[KafkaProducer]:
    """
    Create a Kafka producer with retry logic.
    
    Args:
        retries: Number of connection retries.
        
    Returns:
        KafkaProducer instance or None if connection fails.
    """
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1,
            )
            logger.info(f"Connected to Kafka broker at {KAFKA_BROKER}")
            return producer
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


def on_send_success(record_metadata):
    """Callback for successful message delivery."""
    logger.debug(
        f"Message delivered to {record_metadata.topic} "
        f"partition {record_metadata.partition} "
        f"offset {record_metadata.offset}"
    )


def on_send_error(excp):
    """Callback for failed message delivery."""
    logger.error(f"Message delivery failed: {excp}")


def send_transaction(
    producer: KafkaProducer,
    topic: str,
    transaction: dict
) -> bool:
    """
    Send a single transaction to Kafka.
    
    Args:
        producer: Kafka producer instance.
        topic: Target topic name.
        transaction: Transaction data dictionary.
        
    Returns:
        True if send was initiated successfully.
    """
    try:
        # Use transaction_id as the message key for partitioning
        key = transaction.get('transaction_id', '')
        
        future = producer.send(topic, key=key, value=transaction)
        future.add_callback(on_send_success)
        future.add_errback(on_send_error)
        
        return True
    except KafkaError as e:
        logger.error(f"Failed to send transaction: {e}")
        return False


def stream_transactions(
    producer: KafkaProducer,
    topic: str = KAFKA_TOPIC,
    interval: float = TRANSACTION_INTERVAL,
    max_transactions: int = None,
    fraud_rate: float = 0.05
) -> None:
    """
    Continuously stream transactions to Kafka.
    
    Args:
        producer: Kafka producer instance.
        topic: Target topic name.
        interval: Seconds between transactions.
        max_transactions: Maximum number of transactions (None for infinite).
        fraud_rate: Target fraud rate.
    """
    logger.info(f"Starting transaction stream to topic '{topic}'")
    logger.info(f"Interval: {interval}s, Fraud rate: {fraud_rate:.1%}")
    
    count = 0
    fraud_count = 0
    start_time = datetime.now()
    
    try:
        while max_transactions is None or count < max_transactions:
            # Generate transaction with current timestamp
            transaction = generate_single_transaction(
                base_time=datetime.now(),
                fraud_rate=fraud_rate
            )
            
            # Send to Kafka
            if send_transaction(producer, topic, transaction):
                count += 1
                if transaction['is_fraud']:
                    fraud_count += 1
                
                # Log progress every 10 transactions
                if count % 10 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Sent {count} transactions "
                        f"({fraud_count} fraud, {fraud_count/count:.1%}) "
                        f"Rate: {rate:.1f}/s"
                    )
            
            # Wait before next transaction
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        # Ensure all messages are sent
        producer.flush()
        logger.info(f"Producer stopped. Total sent: {count} transactions")


def main():
    """Main entry point for the producer."""
    logger.info("=" * 50)
    logger.info("FRAUD DETECTION - KAFKA PRODUCER")
    logger.info("=" * 50)
    logger.info(f"Broker: {KAFKA_BROKER}")
    logger.info(f"Topic: {KAFKA_TOPIC}")
    logger.info(f"Interval: {TRANSACTION_INTERVAL}s")
    logger.info("=" * 50)
    
    # Create producer with retry logic
    producer = create_producer()
    
    if producer is None:
        logger.error("Could not create producer. Exiting.")
        sys.exit(1)
    
    try:
        # Start streaming
        stream_transactions(
            producer=producer,
            topic=KAFKA_TOPIC,
            interval=TRANSACTION_INTERVAL,
        )
    finally:
        producer.close()
        logger.info("Producer closed.")


if __name__ == "__main__":
    main()
