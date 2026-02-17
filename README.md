# 🔒 Real-Time Fraud Detection System

A complete end-to-end real-time fraud detection pipeline featuring synthetic transaction generation, Apache Kafka streaming, XGBoost ML model, and a Streamlit dashboard for live monitoring.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.5-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Screenshots](#-screenshots)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## �� Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REAL-TIME FRAUD DETECTION SYSTEM                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│                  │     │                  │     │                  │
│  Transaction     │────▶│  Apache Kafka    │────▶│  ML Consumer     │
│  Generator       │     │  (Message Queue) │     │  (XGBoost)       │
│                  │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
       │                        │                          │
       │                        │                          │
       ▼                        ▼                          ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│                  │     │                  │     │                  │
│  Synthetic Data  │     │  Topic:          │     │  SQLite DB       │
│  (Realistic      │     │  'transactions'  │     │  (Predictions)   │
│   Patterns)      │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                                                           │
                                                           ▼
                                                  ┌──────────────────┐
                                                  │                  │
                                                  │  Streamlit       │
                                                  │  Dashboard       │
                                                  │  (Real-time UI)  │
                                                  │                  │
                                                  └──────────────────┘

Data Flow:
═══════════════════════════════════════════════════════════════════════════════
[Generator] ──JSON──▶ [Kafka Producer] ──▶ [Kafka Topic] ──▶ [Kafka Consumer]
                                                                    │
                                                          [XGBoost Prediction]
                                                                    │
                                                              [SQLite DB]
                                                                    │
                                                           [Streamlit Dashboard]
```

---

## ✨ Features

### 🎲 Data Generation
- Realistic synthetic credit card transactions
- Configurable fraud rate (~5% default, imbalanced)
- 10 merchant categories with risk-weighted fraud probability
- Log-normal distribution for transaction amounts

### 🤖 Machine Learning
- XGBoost classifier for fraud detection
- SMOTE for handling class imbalance
- Comprehensive metrics: Precision, Recall, F1, ROC-AUC
- Model persistence with joblib

### 📡 Streaming Pipeline
- Apache Kafka for reliable message transport
- Configurable transaction throughput
- Automatic retry and error handling
- Consumer group support for scalability

### 📊 Real-Time Dashboard
- Auto-refreshing every 3 seconds
- Live transaction feed
- Fraud rate gauge with color zones
- Transaction volume time series
- Merchant category analysis
- High-risk alert banners (probability > 85%)

---

## 📁 Project Structure

```
real-time-fraud-detection/
├── data/
│   ├── generate_transactions.py   # Synthetic data generator
│   └── predictions.db             # SQLite database (created at runtime)
├── kafka/
│   ├── producer.py                # Kafka transaction producer
│   └── consumer.py                # Kafka consumer with ML inference
├── model/
│   ├── train_model.py             # XGBoost model training
│   └── fraud_model.pkl            # Trained model (created after training)
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── docker-compose.yml             # Kafka + Zookeeper setup
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 📋 Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose** (for Kafka)
- **pip** (Python package manager)

---

## 🚀 Installation

### 1. Clone or Navigate to Project

```bash
cd ~/real-time-fraud-detection
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Kafka Infrastructure

```bash
docker-compose up -d
```

Wait ~30 seconds for services to be ready. Verify with:

```bash
docker-compose ps
```

All services should show "Up" status.

### 5. Train the ML Model

```bash
python model/train_model.py
```

This will:
- Generate 50,000 synthetic transactions
- Train an XGBoost model with SMOTE
- Print evaluation metrics
- Save `fraud_model.pkl` to the model directory

---

## 🎯 Usage

### Running the Complete Pipeline

Open **4 terminal windows** and run each component:

#### Terminal 1: Kafka Producer
```bash
cd ~/real-time-fraud-detection
source venv/bin/activate
python kafka/producer.py
```

#### Terminal 2: Kafka Consumer
```bash
cd ~/real-time-fraud-detection
source venv/bin/activate
python kafka/consumer.py
```

#### Terminal 3: Streamlit Dashboard
```bash
cd ~/real-time-fraud-detection
source venv/bin/activate
streamlit run dashboard/app.py
```

#### Terminal 4: Kafka UI (Optional)
Access the Kafka UI at: http://localhost:8080

### Dashboard Access

Open your browser to: **http://localhost:8501**

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BROKER` | `localhost:9092` | Kafka broker address |
| `KAFKA_TOPIC` | `transactions` | Kafka topic name |
| `TRANSACTION_INTERVAL` | `1.0` | Seconds between transactions |
| `CONSUMER_GROUP` | `fraud-detection-group` | Kafka consumer group |
| `DB_PATH` | `data/predictions.db` | SQLite database path |
| `MODEL_PATH` | `model/fraud_model.pkl` | Trained model path |
| `REFRESH_INTERVAL` | `3` | Dashboard refresh (seconds) |

### Example with Custom Config

```bash
KAFKA_BROKER=localhost:9092 \
TRANSACTION_INTERVAL=0.5 \
python kafka/producer.py
```

---

## 📸 Screenshots

### Dashboard Overview
*[Add screenshot of main dashboard here]*

### Fraud Detection Metrics
*[Add screenshot of metrics panel here]*

### Live Transaction Feed
*[Add screenshot of transaction table here]*

### High-Risk Alerts
*[Add screenshot of alert banner here]*

---

## 🔧 Troubleshooting

### Kafka Connection Issues

```bash
# Check if Kafka is running
docker-compose ps

# View Kafka logs
docker-compose logs kafka

# Restart services
docker-compose restart
```

### Model Not Found Error

```bash
# Ensure model is trained
python model/train_model.py

# Verify model exists
ls -la model/fraud_model.pkl
```

### Database Locked Error

```bash
# Remove stale database
rm data/predictions.db

# Restart consumer
python kafka/consumer.py
```

### Port Already in Use

```bash
# Find process using port
lsof -i :9092  # Kafka
lsof -i :8501  # Streamlit

# Kill process
kill -9 <PID>
```

---

## 🛑 Stopping the System

```bash
# Stop all Docker containers
docker-compose down

# Deactivate virtual environment
deactivate
```

---

## 📈 Performance Tuning

### Increase Throughput

```bash
# Faster transaction generation
TRANSACTION_INTERVAL=0.1 python kafka/producer.py
```

### Multiple Consumers

Run multiple consumer instances (they share the workload via consumer groups):

```bash
# Terminal 2a
CONSUMER_GROUP=fraud-group python kafka/consumer.py

# Terminal 2b (same group = load balanced)
CONSUMER_GROUP=fraud-group python kafka/consumer.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Apache Kafka](https://kafka.apache.org/) for reliable streaming
- [XGBoost](https://xgboost.readthedocs.io/) for powerful ML
- [Streamlit](https://streamlit.io/) for rapid dashboard development
- [Plotly](https://plotly.com/) for interactive visualizations

---

**Built with ❤️ for real-time fraud detection**
