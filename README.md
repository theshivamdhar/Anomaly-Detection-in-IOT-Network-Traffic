<h1 align="center">IoT Intrusion Detection System</h1>

<p align="center">
  <b>Machine Learning-Based Network Anomaly Detection for IoT Environments</b><br>
  <i>Using Random Forest, Decision Tree & XGBoost — with a Streamlit Dashboard</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Academic-green" />
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#dataset-instructions">Dataset</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-run">How to Run</a> •
  <a href="#results">Results</a> •
  <a href="#screenshots">Screenshots</a>
</p>

---

## About the Project

The number of IoT devices is growing rapidly — smart homes, industrial sensors, wearables — and every connected device is a potential entry point for cyber attacks. Traditional firewalls rely on known attack signatures and fail against zero-day threats.

This project builds an **Intrusion Detection System (IDS)** that uses machine learning to **learn what normal network traffic looks like** and automatically flag anomalies — detecting attacks it has never seen before.

> **B.Tech CSE Minor Project — 2026**

### Why This Matters

| Question | Answer |
|----------|--------|
| What is the risk? | IoT devices lack built-in security — easy targets for botnets, DDoS, and data theft |
| Why not just use firewalls? | Signature-based tools only catch known attacks. New/zero-day threats pass through |
| Why machine learning? | ML learns traffic patterns from data, detecting anomalies it has never been trained on |

---

## Features

- **Multi-class Attack Detection** — Classifies traffic as Normal, DoS, Port Scan, or Data Exfiltration
- **Binary & Multiclass Modes** — Switch between binary (Normal vs Attack) and detailed multiclass detection
- **3 ML Models Compared** — Random Forest, Decision Tree, and XGBoost with full performance comparison
- **Rich Evaluation Suite** — Confusion matrices, ROC curves, Precision-Recall curves, feature importance plots
- **5-Fold Cross-Validation** — Robust training with stratified CV to prevent overfitting
- **Streamlit Web Dashboard** — Interactive UI for real-time predictions, CSV upload, and visual exploration
- **Feature Engineering** — 4 derived features (`byte_ratio`, `packet_rate`, `src_dst_ratio`, `error_flag_interact`)
- **Permutation Importance** — Model-agnostic feature importance analysis

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| ML Models | Scikit-learn (Random Forest, Decision Tree), XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web Dashboard | Streamlit |
| Serialization | Joblib |
| Dataset | Edge-IIoT Dataset / Synthetic IoT Traffic |

---

## Project Structure

```
IOT_Minor/
│
├── data/                          # ⚠ NOT included in repo (see Dataset section)
│   ├── ML-EdgeIIoT-dataset.csv    #   Main Edge-IIoT dataset (~78 MB)
│   ├── dataset.csv                #   Processed/synthetic dataset
│   └── iot_demo_dataset.csv       #   Demo dataset
│
├── models/                        # ⚠ NOT included in repo (generated during training)
│   ├── random_forest.pkl          #   Trained Random Forest model
│   ├── decision_tree.pkl          #   Trained Decision Tree model
│   ├── xgboost.pkl                #   Trained XGBoost model
│   ├── scaler_*.pkl               #   Fitted scalers
│   ├── label_encoder_*.pkl        #   Label encoders
│   ├── confusion_matrix_*.png     #   Confusion matrix plots
│   ├── roc_curve_*.png            #   ROC curve plots
│   ├── pr_curve_*.png             #   Precision-Recall curve plots
│   └── feature_importance_*.png   #   Feature importance plots
│
├── app.py                         # Streamlit dashboard application
├── train_rf.py                    # Model training & evaluation script
├── generate_dataset.py            # Synthetic dataset generator
├── utils.py                       # Utility functions (preprocessing, metrics)
├── requirements.txt               # Python dependencies
├── Synopsis_Minor_Project.md      # Project synopsis document
└── README.md                      # You are here
```

---

## Dataset Instructions

> **⚠ Important:** The dataset and trained models are **not included** in this repository due to GitHub's 100 MB file size limit.

### Option A — Download the Dataset

Download from Google Drive:

> [Anomaly Detection in IoT Network Traffic](https://drive.google.com/drive/folders/1WSKC48AaI9dDUKR2y7_8LZ8FwrAW7c7n?usp=drive_link)

After downloading, place the CSV file inside the `data/` folder:

```
IOT_Minor/
└── data/
    └── ML-EdgeIIoT-dataset.csv    ← Place the file here
```

### Option B — Contact Me

If the download link is unavailable, reach out via email: **theshivamdhar@gmail.com**

### Option C — Generate a Synthetic Dataset

To test the project without the real dataset, generate a synthetic one:

```bash
python generate_dataset.py
```

This creates `data/dataset.csv` with 10,000 synthetic network flows — sufficient to train and evaluate all models.

### About the Dataset

The **Edge-IIoT dataset** contains real-world IoT network traffic with labelled attack categories.

| Property | Details |
|----------|---------|
| Source | Edge-IIoT (IoT/IIoT cybersecurity dataset) |
| Size | ~78 MB |
| Features | 14 base features + 4 engineered features |
| Classes | Normal, DoS, Port Scan, Data Exfiltration |
| Modes | Binary (Normal vs Attack) and Multiclass |

---

## Installation

### Prerequisites

- **Python 3.10 or higher** — [Download Python](https://www.python.org/downloads/)
- **Git** — [Download Git](https://git-scm.com/downloads)

### Step-by-Step Setup

**1. Clone the repository**

```bash
git clone https://github.com/theshivamdhar/Anomaly-Detection-in-IOT-Network-Traffic.git
cd Anomaly-Detection-in-IOT-Network-Traffic
```

**2. Create a virtual environment** (recommended)

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Follow the [Dataset Instructions](#dataset-instructions) above to download, request, or generate the dataset.

---

## How to Run

### Step 1 — Train the Models

Train all three models (Random Forest, Decision Tree, XGBoost) on the dataset.

**Binary classification** (Normal vs Attack):

```bash
python train_rf.py --data data/dataset.csv --mode binary
```

**Multiclass classification** (Normal, DoS, Port Scan, Exfiltration):

```bash
python train_rf.py --data data/dataset.csv --mode multiclass
```

> The script preprocesses data, engineers features, trains all models, runs cross-validation, generates evaluation plots, and saves everything to the `models/` folder.

### Step 2 — Launch the Dashboard

```bash
streamlit run app.py
```

This opens a web dashboard in your browser where you can:
- Select a trained model for predictions
- Upload CSV files for batch prediction
- View confusion matrices, ROC curves, and feature importance
- Explore model comparison metrics

### Quick Reference (All Commands)

```bash
# 1. Generate synthetic dataset (if needed)
python generate_dataset.py

# 2. Train models — binary mode
python train_rf.py --data data/dataset.csv --mode binary

# 3. Train models — multiclass mode
python train_rf.py --data data/dataset.csv --mode multiclass

# 4. Launch dashboard
streamlit run app.py
```

---

## Model Training Details

### Pipeline Flow

```
Raw CSV Data
     ↓
Preprocessing (clean, deduplicate, impute missing values)
     ↓
Feature Engineering (4 derived features added)
     ↓
Train/Test Split (80/20, stratified)
     ↓
Feature Scaling (StandardScaler — fitted on train only)
     ↓
Model Training (Random Forest + Decision Tree + XGBoost)
     ↓
5-Fold Stratified Cross-Validation
     ↓
Evaluation (metrics, confusion matrix, ROC, PR curves)
     ↓
Save Models & Plots → models/ folder
```

### Models Used

| Model | Configuration | Strengths |
|-------|--------------|-----------|
| **Random Forest** | 100 trees, balanced class weights | Robust, handles noise well, provides feature importance |
| **Decision Tree** | Max depth 15 | Fast, interpretable, good baseline |
| **XGBoost** | Gradient boosted trees | High accuracy, handles class imbalance, regularized |

### Engineered Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `byte_ratio` | dst_bytes / (src_bytes + 1) | Data flow direction — high in exfiltration |
| `packet_rate` | packet_size / (duration + 0.01) | Throughput — spikes during DoS |
| `src_dst_ratio` | src_bytes / (dst_bytes + 1) | Inverse flow asymmetry |
| `error_flag_interact` | error_rate × wrong_fragment | Compound anomaly signal |

### Saved Outputs

After training, the `models/` folder will contain:
- Trained model files (`.pkl`) — Random Forest, Decision Tree, XGBoost
- Scalers and label encoders (`.pkl`)
- Confusion matrix heatmaps (`.png`)
- ROC curves (`.png`)
- Precision-Recall curves (`.png`)
- Feature importance plots (`.png`)
- Cross-validation results (`.csv`)
- Model comparison table (`.csv`)

---

## Results

### Performance Summary (Binary Mode)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | ~0.91 | ~0.91 | ~0.91 | ~0.91 | ~0.99 |
| Decision Tree | ~0.71 | ~0.87 | ~0.71 | ~0.72 | ~0.95 |
| XGBoost | ~0.92 | ~0.93 | ~0.92 | ~0.93 | ~0.99 |

### Key Insights

- **Random Forest** and **XGBoost** consistently outperform Decision Tree
- Most important features: `connection_rate`, `packet_rate`, `dst_bytes`, `srv_count`
- XGBoost handles class imbalance effectively with built-in regularization

### Attack Signatures Learned

| Attack Type | Key Pattern |
|-------------|-------------|
| **DoS** | Extremely high connection rate, large packets, short duration |
| **Port Scan** | Very short duration, many services probed, tiny packets |
| **Data Exfiltration** | Very high outbound bytes, long duration, moderate connection rate |

---

## Screenshots

| Screenshot | Description |
|------------|-------------|
| <img src="assets/dashboard.png" width="600"> | **Streamlit dashboard** — Overview of Traffic and Model Performance |
| <img src="assets/traffic_distribution.png" width="600"> | **Traffic Distribution** — Breakdown of network traffic across categories |
| <img src="assets/confusion_matrix.png" width="600"> | **Confusion Matrix** — Visualization of actual vs predicted attack classes |
| <img src="assets/roc_curve.png" width="600"> | **ROC Curve** — TPR vs FPR analysis across multiple threat classes |
| <img src="assets/precision_recall.png" width="600"> | **Precision-Recall Curve** — Evaluation of model's detection quality |

---

## Limitations

- **Dataset not included** — Must be downloaded separately or generated synthetically due to GitHub size limits
- **Synthetic data** — Generated distributions approximate real traffic but do not replicate it exactly
- **No temporal modelling** — Each flow is classified independently; slow or sequential attacks may be missed
- **Batch processing only** — The pipeline loads all data into memory; production use requires streaming integration
- **Global explainability only** — Feature importance is global, not per-sample; SHAP integration would improve interpretability

---

## Future Improvements

- **Real-time packet capture** — Integration with Scapy or tcpdump for live traffic analysis
- **Deep Learning models** — LSTM or Transformer architectures for sequential traffic patterns
- **API deployment** — REST API via FastAPI or Flask for production-grade predictions
- **Containerization** — Docker support for reproducible deployment
- **SHAP explainability** — Per-prediction feature attribution for analyst trust
- **Real-world validation** — Testing on BoT-IoT, IoT-23, CIC-IoT-2023 datasets
- **Federated learning** — Privacy-preserving distributed training across IoT networks

---

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Meidan, Y., et al. (2018). N-BaIoT: Network-Based Detection of IoT Botnet Attacks. *IEEE Pervasive Computing*.
- Koroniotis, N., et al. (2019). Towards the Development of Realistic Botnet Dataset in the IoT. *Future Generation Computer Systems*.
- Kolias, C., et al. (2017). DDoS in the IoT: Mirai and Other Botnets. *IEEE Computer*.

---

## Author

| | |
|---|---|
| **Name** | Shivam Dhar |
| **Program** | B.Tech CSE |
| **GitHub** | [@theshivamdhar](https://github.com/theshivamdhar) |
| **Email** | theshivamdhar@gmail.com |

---

<p align="center"><i>B.Tech Minor Project — Academic use only — 2026</i></p>
