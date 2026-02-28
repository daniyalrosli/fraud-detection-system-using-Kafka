Real-Time Fraud Detection System

A complete end-to-end real-time fraud detection pipeline featuring synthetic transaction generation, Apache Kafka streaming, XGBoost ML model, and a Streamlit dashboard for live monitoring.



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
