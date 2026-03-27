# IoT Intrusion Detection System

A machine learning pipeline that detects cyber attacks in IoT network traffic using Random Forest classification.

> B.Tech Minor Project -- 2026

---

## 1. The Problem

IoT devices are multiplying fast -- smart homes, wearables, industrial sensors. Every device is a potential entry point for attackers. Traditional security tools check packets against known attack signatures, but they cannot catch new or unknown threats.

We need a system that **learns what normal traffic looks like** and flags anything that deviates.

### Why this matters

| Question | Answer |
| -------- | ------ |
| What is the risk? | IoT devices lack built-in security and are easy targets for botnets, DDoS, and data theft |
| Why not use firewalls? | Signature-based tools only catch known attacks. Zero-day threats pass through undetected |
| Why machine learning? | ML learns traffic patterns from data, so it can detect anomalies it has never seen before |

---

## 2. The Solution

This project is a complete intrusion detection pipeline with four components:

- **Synthetic Dataset Generator** -- creates realistic IoT traffic with labelled attack types
- **Random Forest Classifier** -- learns to distinguish normal traffic from three attack categories
- **Evaluation Pipeline** -- measures performance with multiple metrics, curves, and visualisations
- **Streamlit Dashboard** -- optional web UI for interactive predictions and exploration

### Pipeline Flow

```text
Raw CSV Data
     ↓
Preprocessing (clean, deduplicate, impute)
     ↓
Feature Engineering (4 derived features)
     ↓
Train/Test Split (80/20, stratified)
     ↓
Feature Scaling (StandardScaler)
     ↓
Model Training (Random Forest + Decision Tree)
     ↓
Evaluation (metrics, confusion matrix, ROC, PR curves)
     ↓
Prediction Output / Dashboard
```

---

## 3. System Architecture

### Data Layer

- Synthetic dataset with 10,000 network flows across 4 traffic classes
- 14 base features with Gaussian noise injection for realism
- Configurable class imbalance (50% normal, 50% attacks)

### Processing Layer

- Duplicate removal and missing value imputation
- 4 engineered features: `byte_ratio`, `packet_rate`, `src_dst_ratio`, `error_flag_interact`
- StandardScaler fitted on training data only

### Modelling Layer

- **Random Forest** -- 100 trees, balanced class weights, parallelised training
- **Decision Tree** -- max depth 15, used as a baseline for comparison
- 5-fold stratified cross-validation on training data

### Evaluation Layer

- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion matrix heatmaps
- ROC and Precision-Recall curves
- Gini-based and permutation-based feature importance

### Interface Layer

- Streamlit web app with model selection, CSV upload, and interactive visualisations
- All trained models and plots saved to `models/` directory

---

## 4. Dataset Design

### Why synthetic?

Real IoT datasets are large, messy, and require hours of preprocessing. A synthetic dataset gives us **controlled ground truth** -- we know exactly which rows are attacks because we generated them.

### How attacks are simulated

Each class is generated with distinct statistical distributions. For example, DoS traffic has extremely high connection rates, while port scans have very short durations and probe many services.

Gaussian noise (8% of feature std) is added to all continuous features to prevent trivially perfect classification.

### Traffic Classes

| Class | Share | Key Behaviour |
| ----- | ----- | ------------- |
| Normal | 50% | Moderate packet sizes, regular durations, low error rates |
| DoS | 20% | High packet sizes, burst connections, elevated errors |
| Port Scan | 15% | Tiny packets, very short durations, many services probed |
| Data Exfiltration | 15% | Large outbound bytes, long durations |

### Feature Table

| Feature | What It Represents | Why It Helps Detect Attacks |
| ------- | ------------------ | --------------------------- |
| `packet_size` | Network packet size in bytes | DoS sends unusually large packets |
| `duration` | Connection length in seconds | Port scans are extremely short |
| `src_bytes` | Bytes from source | Asymmetry reveals abnormal flow |
| `dst_bytes` | Bytes from destination | Very high in data exfiltration |
| `wrong_fragment` | Incorrect fragment count | Elevated in crafted packet attacks |
| `urgent` | Urgent packet count | Rare in normal traffic |
| `count` | Connections in time window | Burst connections signal DoS |
| `srv_count` | Distinct services contacted | High values indicate port scanning |
| `protocol_type` | TCP / UDP / ICMP (encoded) | Attacks prefer certain protocols |
| `connection_rate` | Connections per second | Extremely high in DoS |
| `error_rate` | Fraction of errors | Elevated during malformed attacks |
| `flag` | Connection status flag | Abnormal flags mean incomplete handshakes |
| `land` | Source equals destination | Land attacks set this to 1 |
| `logged_in` | Successful login flag | Attacks often fail to authenticate |

### Engineered Features

| Feature | Formula | What It Captures |
| ------- | ------- | ---------------- |
| `byte_ratio` | dst_bytes / (src_bytes + 1) | Data flow direction -- high in exfiltration |
| `packet_rate` | packet_size / (duration + 0.01) | Throughput -- spikes during DoS |
| `src_dst_ratio` | src_bytes / (dst_bytes + 1) | Inverse flow asymmetry |
| `error_flag_interact` | error_rate x wrong_fragment | Compound anomaly signal |

### Labelling

| Mode | Classes | Encoding |
| ---- | ------- | -------- |
| Binary | Normal vs Attack | 0, 1 |
| Multiclass | Normal, DoS, Port Scan, Exfiltration | 0, 1, 2, 3 |

---

## 5. Methodology

**Step 1 -- Data Cleaning**
Remove duplicates. Fill missing numeric values with medians, categorical values with modes.

**Step 2 -- Feature Engineering**
Derive 4 new features from existing columns. Total features go from 14 to 18.

**Step 3 -- Train/Test Split**
80/20 stratified split. Stratification keeps class proportions consistent in both sets.

**Step 4 -- Feature Scaling**
StandardScaler normalises all features to zero mean and unit variance. Fitted on train only.

**Step 5 -- Model Training**
Random Forest (100 trees, balanced weights) and Decision Tree (depth 15) trained on scaled data.

**Step 6 -- Cross-Validation**
5-fold stratified CV on training set. Reports mean and std for Accuracy, Precision, Recall, F1.

**Step 7 -- Evaluation**
Full metrics on held-out test set. Confusion matrix, ROC curve, PR curve, feature importance plots.

---

## 6. Model Evaluation

### Metrics Explained

| Metric | What It Measures | Why It Matters in IDS |
| ------ | ---------------- | --------------------- |
| Accuracy | Overall correct predictions | Quick check, but misleading if classes are imbalanced |
| Precision | Of predicted attacks, how many are real? | Low precision = too many false alarms |
| Recall | Of real attacks, how many did we catch? | Low recall = missed attacks (dangerous) |
| F1 Score | Balance of Precision and Recall | Single number that captures both concerns |
| ROC-AUC | Discrimination ability across thresholds | Higher = better at separating normal from attack |

### What is a Confusion Matrix?

A confusion matrix shows exactly where the model gets it right and where it makes mistakes. Each row is the actual class, each column is the predicted class. The diagonal shows correct predictions. Off-diagonal entries show specific types of errors -- for instance, how often exfiltration is misclassified as normal traffic.

### Indicative Results (Binary Mode)

| Model | Accuracy | Precision | Recall | F1 | AUC |
| ----- | -------- | --------- | ------ | -- | --- |
| Random Forest | ~0.99 | ~0.99 | ~0.99 | ~0.99 | ~1.00 |
| Decision Tree | ~0.98 | ~0.98 | ~0.98 | ~0.98 | ~0.99 |

### Key takeaway

Random Forest consistently outperforms Decision Tree because it averages 100 decorrelated trees, reducing variance and smoothing the decision boundary.

---

## 7. Key Insights

### Most important features

1. **`connection_rate`** -- strongest signal. DoS attacks produce 85 connections/sec vs 10 for normal.
2. **`packet_rate`** (engineered) -- high throughput in short bursts indicates floods.
3. **`dst_bytes`** -- exfiltration drives outbound bytes to 5000+ vs 450 for normal.
4. **`srv_count`** -- port scans probe 45+ services vs 3 for normal traffic.

### What patterns indicate attacks?

| Attack | Signature Pattern |
| ------ | ----------------- |
| DoS | Extremely high connection rate, large packets, short duration |
| Port Scan | Very short duration, many services probed, small packets |
| Data Exfiltration | Very high outbound bytes, long duration, moderate connection rate |

### Why Random Forest works well here

- Handles mixed feature types (continuous + categorical) natively
- Robust to the 8% noise injection in the synthetic data
- Provides interpretable feature importance scores

---

## 8. Limitations

- **Synthetic data** -- distributions approximate real traffic but do not replicate it exactly. Validate on real datasets before deployment.
- **No temporal modelling** -- each flow is classified independently. Sequential attacks (slow scans) may be missed.
- **Not real-time** -- the pipeline loads all data into memory. Production use requires streaming integration.
- **No per-prediction explainability** -- feature importance is global, not per-sample. SHAP integration is needed for analyst trust.

---

## 9. Future Scope

- Evaluate on real-world IoT datasets (BoT-IoT, IoT-23, CIC-IoT-2023)
- Add LSTM or Transformer models for sequential traffic patterns
- Integrate with Scapy or tcpdump for real-time packet capture
- Add SHAP explainability for per-prediction feature attribution
- Containerise with Docker and expose predictions via FastAPI
- Explore federated learning for privacy-preserving distributed training

---

## 10. Tech Stack

| Layer | Technology |
| ----- | ---------- |
| Language | Python 3.14 |
| ML Models | scikit-learn (Random Forest, Decision Tree) |
| Data | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Web UI | Streamlit |
| Serialisation | joblib |
| Dataset | Synthetic IoT Traffic (10,000 flows) |

---

## 11. How to Run

### Generate dataset

```bash
python generate_dataset.py
```

### Train models (binary)

```bash
python train_rf.py --data data/dataset.csv --mode binary
```

### Train models (multiclass)

```bash
python train_rf.py --data data/dataset.csv --mode multiclass
```

### Launch dashboard

```bash
python -m streamlit run app.py
```

---

## Project Structure

```text
IOT_Minor/
|-- data/
|   +-- dataset.csv
|-- models/
|   |-- random_forest.pkl
|   |-- decision_tree.pkl
|   |-- scaler_rf_*.pkl
|   |-- confusion_matrix_*.png
|   |-- roc_curve_*.png
|   |-- pr_curve_*.png
|   |-- feature_importance_*.png
|   +-- permutation_importance_*.png
|-- generate_dataset.py
|-- train_rf.py
|-- utils.py
|-- app.py
|-- requirements.txt
+-- README.md
```

---

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Meidan, Y., et al. (2018). N-BaIoT: Network-Based Detection of IoT Botnet Attacks. *IEEE Pervasive Computing*.
- Koroniotis, N., et al. (2019). Towards the Development of Realistic Botnet Dataset in the IoT. *Future Generation Computer Systems*.
- Kolias, C., et al. (2017). DDoS in the IoT: Mirai and Other Botnets. *IEEE Computer*.

---

*B.Tech Minor Project -- Academic use only.*
