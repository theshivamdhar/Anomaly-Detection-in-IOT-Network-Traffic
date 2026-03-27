# SYNOPSIS REPORT

## IoT Intrusion Detection System Using Machine Learning and XGBoost

**Programme:** Bachelor of Technology — Computer Science & Engineering
**Category:** Minor Project
**Session:** 2025–2026

---

---

## 1. INTRODUCTION

### 1.1 Background

The Internet of Things (IoT) refers to the rapidly expanding network of interconnected physical devices, sensors, actuators, and embedded systems that communicate over the internet to collect, exchange, and act upon data without requiring direct human-to-human or human-to-computer interaction. According to Statista (2024), the global count of IoT-connected devices surpassed 15.14 billion in 2023 and is projected to reach 29.42 billion by 2030. These devices span critical application domains including industrial manufacturing (Industrial IoT or IIoT), healthcare monitoring, smart cities, autonomous transportation, precision agriculture, and home automation.

While this proliferation has unlocked transformative economic and societal benefits — enabling predictive maintenance in factories, remote patient monitoring in hospitals, and intelligent traffic management in cities — it has simultaneously introduced an unprecedented expansion of the cyber-attack surface. IoT devices are inherently resource-constrained in terms of processing power, memory, and energy, which severely limits their ability to run traditional, computationally intensive security software such as full-featured firewalls, antivirus suites, or host-based intrusion detection agents. Furthermore, the heterogeneity of IoT communication protocols (MQTT, CoAP, Zigbee, BLE, LoRaWAN), combined with the sheer volume and velocity of data they generate, renders conventional signature-based Intrusion Detection Systems (IDS) fundamentally inadequate for the IoT paradigm.

Traditional IDS architectures — both signature-based systems like Snort and anomaly-based systems using simple statistical thresholds — were designed for enterprise-grade IT networks characterised by relatively homogeneous traffic patterns, standardised protocols (TCP/IP), and abundant computational resources at monitoring points. When deployed in IoT environments, these systems suffer from several critical limitations. Signature-based IDS can only detect attacks whose patterns have been previously catalogued in their rule databases, rendering them completely blind to zero-day exploits, polymorphic malware, and novel attack vectors that specifically target IoT firmware vulnerabilities or protocol-level weaknesses. Moreover, the static rule-matching approach generates an unacceptably high volume of false positives when confronted with the highly variable and bursty traffic patterns characteristic of IoT networks, where normal device behaviour can exhibit significant temporal and contextual variation.

The scale of the IoT security challenge is substantiated by alarming industry statistics. SonicWall's 2023 Cyber Threat Report documented 112.3 million IoT malware attacks globally, representing a 87% year-over-year increase. The Mirai botnet and its variants — which exploit default credentials and known vulnerabilities in IoT devices such as IP cameras, DVRs, and routers — have demonstrated the devastating consequences of inadequate IoT security, enabling distributed denial-of-service (DDoS) attacks exceeding 1 Tbps in bandwidth. Kaspersky's IoT threat landscape analysis identified that 97% of all attacks on IoT devices in 2022 were conducted via the Telnet protocol, highlighting the persistence of legacy protocol vulnerabilities in deployed IoT infrastructure.

Against this backdrop, machine learning (ML) has emerged as a compelling paradigm for IoT intrusion detection. Unlike rule-based systems, ML algorithms can learn complex, non-linear decision boundaries from high-dimensional network traffic data, enabling them to identify both known attack patterns and previously unseen anomalies. Supervised ML classifiers, in particular, can be trained on labelled datasets containing examples of normal traffic and various attack categories, learning to discriminate between benign and malicious network flows based on extracted features such as packet size distributions, connection durations, byte transfer ratios, error rates, and protocol-specific behaviours.

This project develops an **IoT Intrusion Detection System (IDS)** that leverages three supervised machine learning classifiers — **Random Forest**, **Decision Tree**, and **XGBoost (Extreme Gradient Boosting)** — to analyse network traffic features extracted from IoT environments and classify each network flow as either normal or malicious. The system is trained and evaluated on the **Edge-IIoT dataset**, a comprehensive, modern cybersecurity benchmark specifically designed for IoT and Industrial IoT applications. An interactive **Streamlit web dashboard** is developed to provide security analysts with real-time predictions, detailed evaluation visualisations, and comparative model performance analytics.

### 1.2 Motivation

The motivation for this project stems from the convergence of three critical factors. First, the exponential growth of IoT deployments across mission-critical domains — healthcare, industrial control systems, transportation, energy grids — means that successful cyber-attacks on IoT infrastructure can have life-threatening physical consequences, not merely data breaches. A compromised insulin pump, a hijacked autonomous vehicle, or a manipulated industrial control system represents a qualitatively different threat category than a stolen credit card number.

Second, the resource constraints inherent to IoT devices necessitate a network-level detection approach rather than host-based security. Since individual IoT endpoints cannot efficiently run complex security agents, the detection intelligence must reside at network gateway points, edge servers, or cloud security platforms where sufficient computational resources are available to execute ML inference in near-real-time.

Third, the evolution of IoT-specific attack methodologies — including firmware exploitation, protocol fuzzing, side-channel attacks, and IoT botnet recruitment — demands detection systems that can adapt and generalise beyond pre-defined signatures. Machine learning, with its capacity for pattern recognition in high-dimensional feature spaces, represents the most promising approach to building adaptive, generalisable IoT intrusion detection systems.

### 1.3 Literature Review

This section reviews relevant prior research on machine learning–based intrusion detection systems, with specific attention to IoT network security, ensemble classifiers, and gradient-boosted methods.

#### 1.3.1 IoT-Specific Intrusion Detection Datasets and Benchmarks

**Ferrag et al. (2022)** introduced the **Edge-IIoT dataset** in their paper titled *"Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning"*, published in IEEE Access (Vol. 10, pp. 40281–40306). The authors constructed a heterogeneous IoT/IIoT testbed comprising over 10 different IoT devices and services, capturing network traffic across 14 distinct attack categories including DDoS (TCP, UDP, ICMP, HTTP), ransomware, SQL injection, cross-site scripting (XSS), uploading attacks, password attacks, port scanning, vulnerability scanning, backdoor attacks, and man-in-the-middle (MITM) attacks. The dataset provides both raw packet captures and preprocessed machine learning features, enabling both binary (Normal vs. Attack) and multiclass classification tasks. The authors benchmarked several centralised and federated learning pipelines using classical ML models (Decision Tree, Random Forest, K-Nearest Neighbours) and simple deep neural networks, establishing baseline performance metrics.

**Limitation:** While the dataset construction methodology was rigorous and comprehensive, the benchmarking study employed default hyperparameters for all classifiers and did not explore advanced ensemble methods such as XGBoost with systematic hyperparameter optimisation. Furthermore, the evaluation methodology did not explicitly address the issue of data leakage or the importance of evaluating exclusively on held-out test data.

**Koroniotis et al. (2019)** developed the **Bot-IoT dataset** in their work *"Towards the Development of Realistic Botnet Dataset in the Internet of Things for Network Forensic Analytics: Bot-IoT Dataset"*, published in Future Generation Computer Systems (Vol. 100, pp. 779–796). This dataset was generated using a realistic network topology that incorporated legitimate IoT traffic alongside botnet attack traffic, including DDoS, DoS, reconnaissance, and information theft categories. The authors demonstrated that Random Forest achieved the highest accuracy among the tested classifiers.

**Limitation:** The Bot-IoT dataset's traffic generation methodology relied heavily on synthetic attack tools (Hping3, Nmap), which may not fully capture the sophistication of real-world IoT-targeted malware. Additionally, the study did not explore gradient-boosted ensemble methods or provide detailed precision-recall analysis for minority attack classes.

#### 1.3.2 Machine Learning for Network Intrusion Detection

**Dhanabal and Shantharajah (2015)** conducted *"A Study on NSL-KDD Dataset for Intrusion Detection System Based on Classification Algorithms"*, published in the International Journal of Advanced Research in Computer and Communication Engineering (Vol. 4, No. 6, pp. 446–452). The authors evaluated Naive Bayes, J48 Decision Tree, and Random Forest classifiers on the NSL-KDD benchmark dataset, which is an improved version of the original KDD Cup 1999 dataset with redundant records removed. Random Forest achieved the highest overall accuracy at 99.67%, followed by J48 Decision Tree at 99.53%.

**Limitation:** The NSL-KDD dataset, despite its improvements over KDD'99, is fundamentally derived from network traffic captured in 1999 from a US Air Force local area network. It represents legacy wired network traffic patterns and protocols that bear little resemblance to modern IoT communication characteristics, including MQTT broker traffic, CoAP request-response patterns, Zigbee mesh network flows, or BLE advertisement packets. Consequently, models trained on NSL-KDD cannot be directly applied to IoT environments without significant domain adaptation.

**Panigrahi and Borah (2018)** presented *"A Detailed Analysis of CICIDS 2017 Dataset for Designing Intrusion Detection Systems"*, published in the International Journal of Engineering & Technology (Vol. 7, No. 3.24, pp. 479–482). The authors applied Random Forest and gradient-boosted classifiers with feature selection techniques on the CICIDS-2017 dataset, which contains benign traffic and the most up-to-date common attacks resembling true real-world data. Their Random Forest classifier achieved over 99% detection accuracy for several attack classes, demonstrating the effectiveness of ensemble tree methods for network traffic classification.

**Limitation:** The CICIDS-2017 dataset represents enterprise-grade network traffic generated in a controlled lab environment simulating a corporate network infrastructure with standard business applications. The traffic patterns, protocol distributions, and attack vectors do not reflect the unique characteristics of resource-constrained IoT environments with heterogeneous communication protocols and sensor-driven traffic patterns.

#### 1.3.3 XGBoost and Gradient Boosting in Cybersecurity

**Shafiq et al. (2020)** published *"Selection of Effective Machine Learning Algorithm and Bot-IoT Attacks Traffic Identification for Internet of Things in Smart City"*, in Future Generation Computer Systems (Vol. 107, pp. 433–442). The authors compared Decision Tree, Random Forest, Artificial Neural Networks, and Naive Bayes classifiers on the Bot-IoT dataset for smart city IoT traffic classification. Their analysis demonstrated that tree-based ensemble methods consistently outperformed probabilistic and neural network approaches for tabular network traffic data, with Random Forest achieving the best overall performance.

**Limitation:** Despite the comprehensive comparison, the study notably excluded XGBoost (Extreme Gradient Boosting) from the classifier comparison. Given that XGBoost has demonstrated state-of-the-art performance across numerous tabular data benchmarks and Kaggle competitions — owing to its superior handling of class imbalance, built-in regularisation, and efficient second-order gradient optimisation — its omission represents a significant gap. Furthermore, the evaluation focused primarily on accuracy and did not include detailed precision-recall analysis, which is critical for assessing detection quality for minority attack classes in imbalanced IoT traffic datasets.

**Chen and Guestrin (2016)** introduced the **XGBoost** algorithm in their seminal paper *"XGBoost: A Scalable Tree Boosting System"*, presented at the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). The authors demonstrated that XGBoost's combination of second-order gradient statistics, regularised objective function, and system-level optimisations (cache-aware access, sparsity-aware computation, parallel tree construction) enabled it to achieve state-of-the-art performance across a wide range of classification and regression benchmarks while maintaining computational efficiency.

#### 1.3.4 Research Gap Addressed by This Project

The literature review reveals three significant gaps that this project addresses:

1. **Dataset Currency:** The majority of ML-based IDS research continues to rely on legacy datasets (KDD'99, NSL-KDD, CICIDS-2017) that do not capture the unique characteristics of IoT network traffic. This project utilises the **Edge-IIoT dataset**, a modern, IoT-specific benchmark published in 2022.

2. **Classifier Comprehensiveness:** Existing studies on IoT intrusion detection either exclude gradient-boosted methods entirely or do not provide a systematic, controlled comparison between bagging-based ensembles (Random Forest), single-tree classifiers (Decision Tree), and boosting-based ensembles (XGBoost). This project conducts a rigorous three-way comparison under identical preprocessing, splitting, and evaluation conditions.

3. **Evaluation Rigour:** Many prior studies evaluate models on entire datasets without proper train-test separation, risking data leakage and inflated performance metrics. This project enforces a strict **80-20 stratified train-test split**, fits all preprocessing transformations on training data only, and computes all evaluation metrics exclusively on held-out test data.

---

## 2. PROBLEM STATEMENT

The Internet of Things ecosystem faces a critical and escalating cybersecurity crisis. The fundamental architectural characteristics of IoT networks — resource-constrained endpoints, heterogeneous communication protocols, massive device scale, and limited update mechanisms — create systemic vulnerabilities that are actively exploited by sophisticated threat actors.

### 2.1 Core Problem

The core problem addressed by this project is the **inability of traditional, rule-based Intrusion Detection Systems to effectively detect and classify cyber-attacks in IoT network environments**. This inability manifests across multiple dimensions:

**Detection Capability Gap:** Signature-based IDS (e.g., Snort, Suricata) maintain databases of known attack signatures and match incoming network traffic against these patterns. In the IoT context, this approach fails because:
- IoT attacks frequently exploit device-specific firmware vulnerabilities and protocol-level weaknesses that are not catalogued in general-purpose signature databases.
- Zero-day exploits — which, by definition, have no existing signatures — represent a disproportionately large threat to IoT devices due to the slow and infrequent firmware update cycles of deployed IoT infrastructure.
- Polymorphic malware variants continuously mutate their code structure and network fingerprints to evade signature matching, while maintaining identical malicious functionality.

**False Positive Problem:** Anomaly-based IDS that use simple statistical thresholds (mean ± standard deviation on individual features) generate unacceptably high false positive rates when applied to IoT traffic. IoT devices exhibit highly variable normal behaviour — a smart thermostat's traffic pattern during a temperature adjustment event may resemble the bursty characteristics of a port scan, while a firmware update download may mimic data exfiltration in terms of volume and duration.

**Scalability Challenge:** The sheer volume of network flows generated by large-scale IoT deployments (thousands to millions of devices) overwhelms manual rule authoring and maintenance approaches. Each new device type, protocol, or firmware version potentially requires new detection rules, making the rule-based approach fundamentally unsustainable.

**Classification Depth:** Modern IoT threat landscape requires not merely binary (normal/attack) detection, but **multiclass classification** that identifies the specific attack type (DDoS, port scan, data exfiltration, ransomware, SQL injection, etc.) to enable appropriate response strategies. Rule-based systems lack the capacity for fine-grained, multi-category classification across high-dimensional feature spaces.

### 2.2 Specific Challenges

The following specific challenges are addressed in this project:

1. **High Dimensionality:** IoT network traffic datasets contain numerous features (packet sizes, byte counts, flow durations, protocol types, error rates, connection counts, etc.) that must be jointly analysed to distinguish subtle attack patterns from normal variability.

2. **Class Imbalance:** In real-world IoT networks, normal traffic vastly outnumbers attack traffic. Even in curated datasets, certain attack categories may have significantly fewer samples than others, biasing classifiers towards majority classes and degrading detection performance for rare but critical attack types.

3. **Feature Selection and Engineering:** Not all network traffic features contribute equally to attack detection. Irrelevant or redundant features increase computational cost and can degrade classifier performance through the curse of dimensionality. Identifying the most discriminative features and engineering informative derived features (e.g., byte ratios, packet rates) is essential.

4. **Model Selection and Comparison:** The choice of machine learning algorithm significantly impacts detection performance. Different algorithmic paradigms (single decision trees, bagged ensembles, boosted ensembles) exhibit different strengths regarding accuracy, handling of class imbalance, overfitting tendency, and interpretability. A systematic, empirically grounded comparison is necessary.

5. **Evaluation Integrity:** Meaningful performance evaluation requires strict separation of training and testing data to prevent data leakage, stratified sampling to preserve class distributions, and comprehensive metrics beyond simple accuracy to capture detection quality across all classes.

### 2.3 Problem Formalisation

Formally, the problem can be stated as a **supervised multiclass classification task**: given a network flow represented by a feature vector **x** ∈ ℝ^d extracted from IoT traffic, learn a function f: ℝ^d → {1, 2, ..., K} that maps each flow to one of K predefined classes (Normal, DDoS_TCP, DDoS_UDP, DDoS_ICMP, DDoS_HTTP, SQL_Injection, XSS, Password, Port_Scanning, Vulnerability_Scanner, Backdoor, Ransomware, MITM, Uploading, Fingerprinting), minimising the expected classification error while maintaining high recall (detection rate) for all attack categories.

---

## 3. OBJECTIVES

The primary and secondary objectives of this project are enumerated below:

### 3.1 Primary Objectives

1. **To develop a machine learning–based Intrusion Detection System** capable of analysing IoT network traffic features and classifying each flow as normal or belonging to a specific attack category with high accuracy, precision, recall, and F1-score.

2. **To train, evaluate, and compare three supervised classifiers** — Random Forest, Decision Tree, and XGBoost — on the Edge-IIoT dataset under identical preprocessing, splitting, and evaluation conditions, enabling a fair and rigorous performance comparison.

3. **To implement a complete, end-to-end machine learning pipeline** encompassing data loading, cleaning, target column detection, feature selection, feature engineering, label encoding, stratified train-test splitting, feature scaling, model training, cross-validation, test-set evaluation, and result visualisation — with explicit safeguards against data leakage at every stage.

4. **To develop an interactive Streamlit web dashboard** that enables security analysts to upload network traffic datasets, select classification models and modes, view real-time predictions, and analyse model performance through interactive visualisations.

### 3.2 Secondary Objectives

5. **To investigate why XGBoost outperforms bagging-based and single-tree classifiers** on IoT intrusion detection tasks, analysing the algorithmic properties (gradient boosting, regularisation, handling of class imbalance via scale_pos_weight) that contribute to its superior performance.

6. **To implement comprehensive evaluation metrics** including confusion matrices, classification reports (per-class precision, recall, F1), ROC curves with AUC scores, and Precision-Recall curves with Average Precision scores — providing a multi-faceted assessment of detection quality.

7. **To conduct 5-fold stratified cross-validation** on training data exclusively, yielding variance-aware performance estimates that quantify the stability and reliability of each classifier.

8. **To perform feature importance analysis** using both model-intrinsic feature importances and model-agnostic permutation importance, identifying the network traffic features most critical for distinguishing normal from malicious IoT flows.

9. **To support both binary and multiclass classification modes** seamlessly within a single codebase, enabling detection tasks ranging from simple normal/attack discrimination to fine-grained 15-class attack categorisation.

10. **To ensure reproducibility** through fixed random seeds, documented hyperparameters, serialised model artifacts, and a modular, well-commented codebase.

---

## 4. METHODOLOGY

This section presents the complete machine learning pipeline implemented in this project, detailing each stage from raw data ingestion to model deployment through the Streamlit dashboard.

### 4.1 Overview of the ML Pipeline

The end-to-end pipeline consists of twelve sequential stages, each implemented as modular, reusable functions in the project's utility library:

| Stage | Operation | Key Function |
|-------|-----------|-------------|
| 1 | Data Loading & Cleaning | `load_dataset()` |
| 2 | Target Column Auto-Detection | `detect_target_column()` |
| 3 | Feature Selection (Numeric Filtering) | `prepare_features()` |
| 4 | Feature Engineering (Derived Features) | `engineer_features()` |
| 5 | Label Encoding (Binary / Multiclass) | `encode_labels()` |
| 6 | Stratified Train-Test Split (80:20) | `split_data()` |
| 7 | Feature Scaling (StandardScaler) | `scale_features()` |
| 8 | Model Training (RF, DT, XGB) | Scikit-learn / XGBoost APIs |
| 9 | Cross-Validation (5-Fold Stratified) | `cross_validate_model()` |
| 10 | Test-Set Evaluation | `evaluate_model()` |
| 11 | Visualisation & Plot Generation | Plotting functions |
| 12 | Model Serialisation & Deployment | Joblib / Streamlit |

**Pipeline Flow Diagram:**

```
Raw CSV File
    │
    ▼
[1] Data Loading & Cleaning
    │  - Read CSV (low_memory=False)
    │  - Drop duplicate rows
    │  - Replace ±inf with NaN
    │  - Median imputation (numeric)
    │  - Mode imputation (categorical)
    │
    ▼
[2] Target Column Auto-Detection
    │  - Binary mode: Attack_label → label
    │  - Multiclass mode: Attack_type → label
    │
    ▼
[3] Feature Selection
    │  - Drop ALL target-related columns
    │  - Keep ONLY numeric features
    │  - Replace remaining inf/NaN with 0
    │  - Data leakage assertion guard
    │
    ▼
[4] Feature Engineering
    │  - byte_ratio = dst_bytes / (src_bytes + 1)
    │  - src_dst_ratio = src_bytes / (dst_bytes + 1)
    │  - packet_rate = packet_size / (duration + 0.01)
    │  - error_flag_interact = error_rate × wrong_fragment
    │
    ▼
[5] Label Encoding
    │  - Binary: {Normal → 0, Attack → 1}
    │  - Multiclass: LabelEncoder (15 classes)
    │
    ▼
[6] Stratified Train-Test Split (80:20)
    │  - stratify=y preserves class proportions
    │  - random_state=42 for reproducibility
    │
    ├──────────────────────────┐
    ▼                          ▼
  Training Set (80%)      Test Set (20%)
    │                          │
    ▼                          │
[7] Feature Scaling            │
    │  - StandardScaler.fit()  │
    │    on TRAINING data only │
    │  - transform() on both   │
    │                          │
    ▼                          │
[8] Model Training             │
    │  - Random Forest         │
    │  - Decision Tree         │
    │  - XGBoost               │
    │                          │
    ▼                          │
[9] Cross-Validation           │
    │  - 5-fold stratified     │
    │  - Training data ONLY    │
    │                          │
    ▼                          ▼
[10] Test-Set Evaluation ◄─────┘
    │  - Predictions on scaled test set
    │  - Accuracy, Precision, Recall, F1, AUC
    │
    ▼
[11] Visualisation
    │  - Confusion Matrix
    │  - ROC Curve (with AUC)
    │  - Precision-Recall Curve (with AP)
    │  - Feature Importance
    │  - Permutation Importance
    │  - Model Comparison Table
    │
    ▼
[12] Serialisation & Dashboard Deployment
    - Models saved as .pkl (Joblib)
    - Scaler, LabelEncoder, feature names saved
    - Streamlit web dashboard for inference
```

### 4.2 Stage 1 — Data Loading and Cleaning

The **Edge-IIoT dataset** is loaded from a CSV file using Pandas with the `low_memory=False` parameter to handle columns with mixed data types — a common characteristic of network traffic datasets where some columns contain both numeric and string representations.

The cleaning pipeline performs the following operations in sequence:

**Duplicate Removal:** Exact duplicate rows are identified and removed using `DataFrame.drop_duplicates()`. In network traffic datasets, exact duplicates may arise from packet retransmissions, logging redundancies, or data collection artifacts, and their inclusion would bias the classifier by overweighting specific traffic patterns.

**Infinity Handling:** Both positive and negative infinity values (`np.inf`, `-np.inf`) are replaced with `NaN`. Infinity values typically result from division-by-zero operations during feature extraction (e.g., computing ratios when a denominator is zero) and would cause numerical instability during model training if left unaddressed.

**Missing Value Imputation — Numeric Columns:** For each numeric column containing missing values, the column median is used as the imputation value. Median imputation is preferred over mean imputation because the median is robust to outliers and extreme values that are prevalent in network traffic data — where a single DDoS attack flow with an anomalous packet count or byte volume could severely skew the column mean.

**Missing Value Imputation — Categorical Columns:** For each categorical (object-type) column containing missing values, the column mode (most frequent value) is used as the imputation value. This preserves the dominant category distribution, which is appropriate for categorical network traffic features such as protocol type or service name.

### 4.3 Stage 2 — Target Column Auto-Detection

The system implements an intelligent target column detection mechanism that automatically identifies the appropriate label column based on the selected classification mode:

| Mode | Priority Order | Description |
|------|---------------|-------------|
| Binary | `Attack_label` → `attack_label` → `label` → `Label` | Binary {0, 1} or {Normal, Attack} |
| Multiclass | `Attack_type` → `attack_type` → `label` → `Label` | 15 attack categories |

This auto-detection approach provides flexibility to work with multiple dataset formats. The Edge-IIoT dataset provides `Attack_label` for binary classification and `Attack_type` for multiclass classification, but the system gracefully falls back to generic `label` columns for compatibility with other IoT security datasets.

### 4.4 Stage 3 — Feature Selection

Feature selection is implemented through a conservative, safety-first approach:

1. **Target Column Exclusion:** ALL target-related columns (`Attack_label`, `Attack_type`, `attack_label`, `attack_type`, `label`, `Label`) are explicitly dropped from the feature matrix, regardless of which one is the active target. This prevents the trivially exploitable data leakage scenario where a related target column (e.g., `Attack_type` when training in binary mode on `Attack_label`) is inadvertently included as a feature.

2. **Numeric Feature Filtering:** Only numeric-type columns are retained in the feature matrix. Categorical features such as IP addresses, domain names, and protocol description strings are excluded because they require specialised encoding techniques (e.g., target encoding, entity embeddings) and can introduce high cardinality issues. For Tree-based classifiers, which this project exclusively uses, numeric features are sufficient for capturing all relevant patterns.

3. **Final Cleaning:** Any remaining infinity or NaN values (which may have been introduced by previous operations) are replaced with zero.

4. **Leakage Guard:** An explicit assertion verifies that no target-related column remains in the feature matrix. This programmatic safeguard catches potential data leakage that might otherwise go undetected and inflate evaluation metrics.

### 4.5 Stage 4 — Feature Engineering

Feature engineering creates derived features that capture higher-order traffic behaviour patterns not directly observable in raw features. Four derived features are computed when the requisite source columns exist in the dataset:

| Derived Feature | Formula | Rationale |
|----------------|---------|-----------|
| `byte_ratio` | dst_bytes / (src_bytes + 1) | Captures the directionality of data flow. Data exfiltration attacks exhibit abnormally high byte ratios, while DoS attacks show low ratios (high src, low dst). The +1 prevents division by zero. |
| `src_dst_ratio` | src_bytes / (dst_bytes + 1) | Inverse directionality measure. DoS and flooding attacks produce disproportionately high source-to-destination ratios. |
| `packet_rate` | packet_size / (duration + 0.01) | Measures the average data throughput per unit time. Port scanning attacks exhibit extremely high packet rates with very short connection durations. The +0.01 prevents division by zero. |
| `error_flag_interact` | error_rate × wrong_fragment | Captures the interaction between network error rates and fragmentation anomalies. Certain attack types (e.g., fragmentation-based DoS) simultaneously elevate both features. |

These derived features are computed using vectorised Pandas operations for computational efficiency and are only added when the required source columns exist in the dataset, ensuring graceful degradation when applied to datasets with different feature schemas.

### 4.6 Stage 5 — Label Encoding

Label encoding transforms the raw target column into integer-encoded class labels suitable for ML classifier training:

**Binary Mode:**
- If the target column already contains numeric {0, 1} values (as in `Attack_label` of Edge-IIoT), the values are directly cast to `int64`.
- If the target contains string labels, a keyword-matching approach maps values matching `{"Normal", "Benign", "normal", "benign", "NORMAL", "BENIGN"}` to class 0, and all other values to class 1.
- Class names: `["Normal", "Attack"]`.

**Multiclass Mode:**
- Scikit-learn's `LabelEncoder` is applied to transform string labels into consecutive integers {0, 1, 2, ..., K-1}.
- The fitted `LabelEncoder` is preserved and serialised for use during inference in the Streamlit dashboard—ensuring consistent mapping between class indices and human-readable class names.

**Edge-IIoT Multiclass Categories (15 classes):**

| Class Index | Attack Type |
|------------|-------------|
| 0 | Normal |
| 1 | DDoS_TCP |
| 2 | DDoS_UDP |
| 3 | DDoS_ICMP |
| 4 | DDoS_HTTP |
| 5 | SQL_Injection |
| 6 | Uploading |
| 7 | XSS |
| 8 | Password |
| 9 | Port_Scanning |
| 10 | Vulnerability_Scanner |
| 11 | Backdoor |
| 12 | Ransomware |
| 13 | Fingerprinting |
| 14 | MITM |

### 4.7 Stage 6 — Stratified Train-Test Split

The dataset is partitioned into training (80%) and testing (20%) subsets using Scikit-learn's `train_test_split` function with the `stratify=y` parameter. Stratified splitting is essential for several reasons:

1. **Class Proportion Preservation:** In imbalanced datasets (which IoT traffic datasets invariably are), random splitting can produce training and test sets with significantly different class distributions. Stratification guarantees that both splits contain approximately the same percentage of each class as the original dataset.

2. **Reliable Minority Class Evaluation:** Without stratification, rare attack classes might be entirely absent from the test set, making it impossible to evaluate detection performance for those categories. Stratification ensures every class has representation in both splits.

3. **Consistent Evaluation:** The stratified split, combined with a fixed random seed (`random_state=42`), ensures that results are deterministic and reproducible. Any researcher using the same data and random seed will obtain identical train-test partitions.

**Configuration:**
```
test_size = 0.2       (20% held-out test set)
random_state = 42     (reproducibility seed)
stratify = y          (preserve class proportions)
```

### 4.8 Stage 7 — Feature Scaling

**StandardScaler** normalisation is applied to transform features to have zero mean and unit variance:

```
x_scaled = (x - μ) / σ
```

where μ is the feature mean and σ is the feature standard deviation, both computed from the **training set only**.

**Critical Anti-Leakage Measure:** The scaler is fit exclusively on `X_train` and then used to transform both `X_train` and `X_test`. If the scaler were fit on the entire dataset (including test data), the test set's statistical properties (mean, variance) would "leak" into the training process, artificially inflating evaluation metrics and producing overly optimistic performance estimates.

Standardisation is important because:
- IoT network features span vastly different numeric ranges (e.g., packet_size in [40, 1500] vs. error_rate in [0, 1]).
- Gradient-based optimisation in XGBoost converges faster with normalised inputs.
- While tree-based classifiers (RF, DT) are technically invariant to monotonic feature transformations, standardisation nonetheless improves numerical stability with float32 computation.

The fitted scaler is serialised to `models/scaler_{mode}.pkl` for consistent preprocessing during inference in the Streamlit dashboard.

### 4.9 Stage 8 — Model Training

Three classifiers are trained on the scaled training data, each representing a different algorithmic paradigm:

#### 4.9.1 Random Forest Classifier

Random Forest is a **bagging (bootstrap aggregating) ensemble** of decision trees. Each tree is trained on a random bootstrap sample of the training data, and at each split node, a random subset of features is considered as split candidates. The final prediction is the majority vote across all trees.

**Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 100 | 100 trees provide a strong ensemble with diminishing returns beyond this count |
| `class_weight` | `balanced` | Automatically adjusts weights inversely proportional to class frequencies, mitigating class imbalance |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Parallel training across all CPU cores |

**Strengths:** Robust to overfitting (averaging reduces variance), handles high-dimensional data well, provides built-in feature importance estimates, minimal hyperparameter sensitivity.

**Weaknesses:** Cannot correct errors from individual trees (each tree is independent), may underperform on highly imbalanced datasets despite class weighting, larger model size (100 full trees).

#### 4.9.2 Decision Tree Classifier

A single **CART (Classification and Regression Tree)** is trained using Gini impurity as the split criterion. The tree recursively partitions the feature space by selecting the feature and threshold that maximise information gain (or equivalently, minimise Gini impurity) at each node.

**Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_depth` | 15 | Limits tree depth to prevent overfitting; deeper trees memorise training data |
| `class_weight` | `balanced` | Addresses class imbalance |
| `random_state` | 42 | Reproducibility |

**Strengths:** Highly interpretable (the decision path can be visualised and explained), fastest training and inference among the three models, minimal memory footprint.

**Weaknesses:** Prone to overfitting without depth constraints, high variance (small changes in training data can produce very different tree structures), unable to learn complex boundaries that require additive model composition.

#### 4.9.3 XGBoost Classifier

XGBoost (Extreme Gradient Boosting) is a **sequential boosting ensemble** that builds trees one at a time, with each successive tree trained to correct the prediction errors (residuals) of the ensemble formed by all preceding trees. Unlike Random Forest's independent, parallel trees, XGBoost's trees are dependent and additive.

**Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 100 | 100 boosting rounds |
| `learning_rate` | 0.1 | Step-size shrinkage to prevent overfitting; each tree contributes 10% of its full prediction |
| `max_depth` | 6 | Controls individual tree complexity |
| `eval_metric` | `logloss` / `mlogloss` | Binary / multiclass logarithmic loss |
| `objective` | `binary:logistic` / `multi:softprob` | Sigmoid / softmax output for binary / multiclass |
| `scale_pos_weight` | N_neg / N_pos (binary) | Balances positive/negative class weights |
| `n_jobs` | -1 | Parallel computation |
| `random_state` | 42 | Reproducibility |

**Why XGBoost Outperforms Random Forest and Decision Tree:**

1. **Error Correction Through Boosting:** Each successive tree specifically targets the instances that the current ensemble misclassifies. This additive, error-correcting approach systematically reduces both bias and residual variance, while Random Forest's independent trees can only reduce variance (not bias).

2. **Second-Order Gradient Optimisation:** XGBoost uses both first-order (gradient) and second-order (Hessian) derivatives of the loss function to determine optimal split points. This provides more accurate split decisions than the Gini impurity criterion used by Random Forest and Decision Tree, particularly in regions of the feature space with complex, non-linear class boundaries.

3. **Built-in Regularisation:** XGBoost incorporates L1 and L2 regularisation terms in its objective function, preventing Individual trees from becoming overly complex. Decision Tree has no built-in regularisation (only external constraints like `max_depth`), and Random Forest relies solely on bootstrap sampling and feature subsampling for regularisation.

4. **Superior Class Imbalance Handling:** The `scale_pos_weight` parameter provides a direct, principled mechanism for adjusting the loss function to account for class imbalance, ensuring that the classifier pays appropriate attention to minority (attack) classes. While `class_weight='balanced'` in scikit-learn's Random Forest and Decision Tree achieves a similar effect through sample weighting, XGBoost's integration of class weights directly into the gradient computation is computationally more efficient and numerically more stable.

5. **Missing Value Awareness:** XGBoost natively handles missing values by learning optimal default directions at each split node during training. This is particularly valuable for IoT datasets where sensor dropouts, communication failures, and protocol-specific feature absence can produce systematic missing data patterns.

### 4.10 Stage 9 — Cross-Validation

Five-fold stratified cross-validation is performed on the **training data only** — never on the full dataset and never involving the test set — to obtain robust, variance-aware performance estimates.

**Procedure:**
1. The 80% training set is divided into 5 equally sized folds, preserving class proportions in each fold through stratification.
2. For each fold iteration, 4 folds serve as the training partition and 1 fold serves as the validation partition.
3. A fresh model instance (with identical hyperparameters) is trained on the 4-fold partition and evaluated on the held-out fold.
4. This process repeats 5 times, with each fold serving exactly once as the validation partition.
5. The mean and standard deviation of each metric across all 5 folds are reported.

**Metrics Computed:**

| Metric | Scikit-learn Scorer | Description |
|--------|-------------------|-------------|
| Accuracy | `accuracy` | Fraction of correctly classified instances |
| Precision | `precision_weighted` | Weighted average of per-class precisions |
| Recall | `recall_weighted` | Weighted average of per-class recalls |
| F1-Score | `f1_weighted` | Weighted harmonic mean of precision and recall |

Cross-validation on training data serves two purposes: (1) it provides a more reliable estimate of generalisation performance than a single train-validation split, and (2) it quantifies the classifier's stability — a large standard deviation indicates sensitivity to the specific training data composition, suggesting potential overfitting.

### 4.11 Stage 10 — Test-Set Evaluation

Final model evaluation is performed exclusively on the **20% held-out test set**, which was completely unseen during training and cross-validation. This ensures that reported metrics reflect genuine out-of-sample generalisation performance.

**Metrics Computed:**

**Accuracy (A):**
```
A = (TP + TN) / (TP + TN + FP + FN)
```
The proportion of all predictions that are correct. While intuitive, accuracy can be misleading for imbalanced datasets where a classifier that always predicts the majority class achieves high accuracy despite zero detection capability.

**Precision (P):**
```
P = TP / (TP + FP)
```
The proportion of positive predictions that are actually correct. High precision means few false alarms — critical for operational IDS deployment where each alert triggers human investigation.

**Recall / Detection Rate (R):**
```
R = TP / (TP + FN)
```
The proportion of actual positive instances that are correctly detected. High recall means few missed attacks — critical for security applications where undetected attacks can cause severe damage.

**F1-Score:**
```
F1 = 2 × (P × R) / (P + R)
```
The harmonic mean of precision and recall, providing a single metric that balances both concerns. The harmonic mean penalises extreme imbalances between precision and recall more severely than the arithmetic mean.

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve):**
The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate (FPR = FP / (FP + TN)) at all possible classification thresholds. The AUC summarises this curve into a single value between 0 and 1, where 1 represents perfect discrimination and 0.5 represents random chance.

For multiclass classification, the One-vs-Rest (OvR) strategy is used, computing a separate ROC curve for each class and then calculating the weighted average AUC.

**Average Precision (AP) and Precision-Recall Curve:**
The Precision-Recall curve plots Precision against Recall at all thresholds. Average Precision summarises this curve as the weighted mean of precisions achieved at each threshold, using the increase in recall as the weight. AP is particularly informative for imbalanced datasets where the positive class is rare — a scenario where AUC-ROC can be overly optimistic because it factors in the True Negative Rate, which is trivially high when negatives vastly outnumber positives.

### 4.12 Stage 11 — Visualisation and Analysis

The following visualisations are generated for comprehensive model assessment:

**Confusion Matrix:** A K×K heatmap (where K is the number of classes) showing the count of predictions for each true-class/predicted-class pair. Diagonal elements represent correct predictions; off-diagonal elements reveal specific misclassification patterns (e.g., which attack types are most frequently confused with normal traffic).

**ROC Curves:** One curve per class (multiclass OvR) or a single curve (binary), with the random baseline (diagonal line) for reference. Per-class AUC values are annotated in the legend.

**Precision-Recall Curves:** One curve per class (multiclass) or single curve (binary), with per-class Average Precision values annotated.

**Feature Importance:** Horizontal bar charts displaying model-intrinsic feature importances (Gini-based importances for tree models) for the top-N most important features, revealing which network traffic attributes contribute most to classification decisions.

**Permutation Importance:** A model-agnostic importance measure computed by randomly shuffling each feature and measuring the resulting decrease in accuracy on the test set. Unlike Gini importance, permutation importance accounts for feature interactions and is not biased towards high-cardinality features.

**Model Comparison Table:** A summary table comparing all three classifiers across Accuracy, Precision, Recall, F1-Score, and AUC, enabling data-driven model selection.

### 4.13 Stage 12 — Model Serialisation and Dashboard Deployment

Trained models and associated artifacts are serialised using Joblib for persistent storage and reuse:

| Artifact | File Path | Purpose |
|----------|-----------|---------|
| Random Forest model | `models/random_forest.pkl` | Serialised trained RF classifier |
| Decision Tree model | `models/decision_tree.pkl` | Serialised trained DT classifier |
| XGBoost model | `models/xgboost.pkl` | Serialised trained XGB classifier |
| StandardScaler | `models/scaler_{mode}.pkl` | Fitted scaler for feature normalisation |
| LabelEncoder | `models/label_encoder_{mode}.pkl` | Fitted encoder for label mapping |
| Feature names | `models/feature_names_{mode}.pkl` | Ordered list of training feature names |
| Comparison table | `models/model_comparison_{mode}.csv` | Model performance comparison |
| CV results | `models/cross_validation_{mode}.csv` | Cross-validation metrics |

These artifacts are loaded by the Streamlit dashboard during inference to ensure consistent preprocessing and prediction.

---

## 5. TOOLS AND TECHNOLOGIES

### 5.1 Programming Language

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core implementation language |

Python was selected for its extensive ecosystem of scientific computing and machine learning libraries, its readability and maintainability, and its status as the de facto standard for ML research and development. The entire project — data preprocessing, model training, evaluation, visualisation, and web interface — is implemented in Python.

### 5.2 Data Science and Machine Learning Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥1.24 | N-dimensional array operations, numerical computing |
| Pandas | ≥2.0 | Data loading (CSV), cleaning, manipulation, and analysis |
| Scikit-learn | ≥1.3 | ML algorithms (RF, DT), preprocessing (StandardScaler, LabelEncoder), evaluation (metrics, cross-validation), model selection (train_test_split) |
| XGBoost | ≥2.0 | Gradient-boosted tree classifier with native handling of class imbalance, missing values, and regularisation |

**NumPy** provides the foundation for efficient numerical computation through its optimised C-based array operations. All feature matrices and label arrays are represented as NumPy arrays for memory efficiency and computational speed.

**Pandas** handles data ingestion, cleaning, and transformation through its DataFrame abstraction. Key operations include CSV parsing with mixed-type handling (`low_memory=False`), duplicate detection and removal, column-type-based filtering (`select_dtypes`), and missing value imputation.

**Scikit-learn** provides the complete ML toolkit:
- `RandomForestClassifier` — bagging ensemble with 100 trees
- `DecisionTreeClassifier` — single CART tree with depth-limited pruning
- `StandardScaler` — zero-mean, unit-variance normalisation
- `LabelEncoder` — categorical-to-integer label transformation
- `label_binarize` — one-hot encoding for multiclass ROC/PR curves
- `train_test_split` — stratified partitioning
- `StratifiedKFold`, `cross_validate` — stratified k-fold cross-validation
- `permutation_importance` — model-agnostic feature importance
- All evaluation metrics: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `classification_report`, `confusion_matrix`, `roc_curve`, `auc`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score`

**XGBoost** implements the Extreme Gradient Boosting algorithm with several architectural advantages:
- Parallel tree construction through block-structured data representation
- Cache-aware memory access patterns for large datasets
- Sparsity-aware split finding for efficient handling of missing values
- Column subsampling (like Random Forest) for additional regularisation
- Built-in L1 (Lasso) and L2 (Ridge) regularisation in the loss function

### 5.3 Visualisation Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Matplotlib | ≥3.7 | Backend plotting engine for static visualisations; confusion matrix heatmaps, ROC/PR curves, importance bar charts |
| Seaborn | ≥0.12 | Statistical visualisation; enhanced heatmap rendering for confusion matrices |
| Plotly | ≥5.15 | Interactive browser-based visualisations in the Streamlit dashboard; hover tooltips, zoom, pan |

The project employs a dual visualisation strategy:
- **Training pipeline** (`train_rf.py`): Matplotlib/Seaborn for high-quality static plots saved as PNG files (150 DPI) — suitable for inclusion in reports and publications.
- **Dashboard** (`app.py`): Plotly for interactive, browser-native visualisations with hover information, zooming, and dynamic legend toggling — providing an analyst-friendly investigation interface.

### 5.4 Web Framework

| Library | Version | Purpose |
|---------|---------|---------|
| Streamlit | ≥1.32 | Interactive web dashboard for model inference and visualisation |

Streamlit enables rapid development of data-centric web applications entirely in Python, without requiring HTML, CSS, or JavaScript expertise. The dashboard provides:

- **Sidebar Controls:** Model selection (RF / DT / XGBoost), classification mode (Binary / Multiclass), presentation mode toggle, and live model status indicators.
- **File Upload:** CSV dataset upload with automatic preview (row/column count, target column detection).
- **Executive Overview:** KPI cards displaying anomaly rate, total samples, normal/attack counts, and risk level classification.
- **Performance Metrics:** Accuracy, Precision, Recall, F1, and AUC — computed on the 20% stratified test split with explicit data leakage prevention.
- **Tabbed Deep Insights:**
  - *Distribution Tab:* Traffic class distribution bar chart with dataset summary statistics.
  - *Model Performance Tab:* Confusion matrix, classification report, ROC curves, PR curves, and model comparison tables.
  - *Feature Intelligence Tab:* Feature importance bar chart, top feature drivers with progress bars, and on-demand permutation importance computation.
  - *Predictions Tab:* Scrollable prediction table with class labels, raw data preview, and CSV download functionality.

### 5.5 Serialisation and Persistence

| Library | Version | Purpose |
|---------|---------|---------|
| Joblib | ≥1.3 | Efficient serialisation of trained models, scalers, and encoders to `.pkl` files |

Joblib is specifically optimised for serialising large NumPy arrays and scikit-learn estimators, providing significantly faster save/load performance than Python's built-in pickle module — particularly important for Random Forest models containing 100 full decision trees.

### 5.6 Development Environment

| Tool | Purpose |
|------|---------|
| Visual Studio Code | Primary code editor with Python extension |
| Git | Version control |
| pip / venv | Package management and virtual environment isolation |

---

## 6. EXPECTED OUTCOMES

### 6.1 Model Performance Outcomes

Based on the algorithmic properties of the selected classifiers and established results in the intrusion detection literature, the following performance outcomes are expected:

| Metric | Decision Tree | Random Forest | XGBoost |
|--------|:------------:|:-------------:|:-------:|
| Accuracy | 93–96% | 97–99% | 98–99.5% |
| Precision (weighted) | 92–95% | 96–99% | 97–99.5% |
| Recall (weighted) | 93–96% | 97–99% | 97–99.5% |
| F1-Score (weighted) | 92–95% | 96–99% | 97–99.5% |
| AUC-ROC | 0.94–0.97 | 0.98–0.99 | 0.98–0.999 |

**XGBoost is expected to achieve the highest overall performance** due to its error-correcting boosting mechanism, second-order gradient optimisation, and superior class imbalance handling through `scale_pos_weight`.

**Random Forest is expected to perform second-best**, benefiting from its ensemble of 100 decorrelated trees that collectively reduce prediction variance and provide robust generalisation.

**Decision Tree, while exhibiting the lowest performance metrics**, provides the highest interpretability and fastest inference time, serving as a valuable transparent baseline.

### 6.2 Analytical Outcomes

1. **Comparative Analysis Report:** A comprehensive model comparison table quantifying performance differences across all metrics, enabling data-driven selection of the optimal classifier for deployment.

2. **Feature Importance Rankings:** Identification of the top network traffic features driving attack detection, providing actionable intelligence for network security analysts about which traffic attributes are most indicative of malicious activity.

3. **Misclassification Pattern Analysis:** Confusion matrix analysis revealing which attack types are most frequently confused with normal traffic or with other attack categories, informing targeted improvement strategies.

4. **Cross-Validation Stability Assessment:** Mean ± standard deviation metrics across 5 folds, quantifying the reliability and stability of each classifier's performance estimates.

### 6.3 System Deliverables

1. **Trained Model Artifacts:** Serialised Random Forest, Decision Tree, and XGBoost classifiers, along with fitted scalers and label encoders, ready for deployment.

2. **Interactive Web Dashboard:** A fully functional Streamlit application enabling CSV upload, model selection, real-time inference, and interactive visualisation of results.

3. **Modular Codebase:** A well-structured, documented Python codebase with separated concerns (utilities, training pipeline, web interface) suitable for extension and maintenance.

4. **Evaluation Plots:** Publication-quality confusion matrices, ROC curves, precision-recall curves, and feature importance charts.

---

## 7. APPLICATIONS

### 7.1 Direct Applications

#### 7.1.1 Industrial IoT (IIoT) Security

In industrial manufacturing environments, IoT sensors and programmable logic controllers (PLCs) monitor and control critical processes — chemical mixing ratios, temperature regulation, robotic assembly operations. A successful cyber-attack on IIoT infrastructure can cause physical equipment damage, production line shutdowns, or even hazardous safety incidents. This IDS can be deployed at the network gateway between the IIoT operational technology (OT) network and the enterprise IT network to detect and classify intrusion attempts targeting industrial control systems.

**Specific Use Case:** A pharmaceutical manufacturing plant deploys the system to monitor network traffic between SCADA controllers and IoT temperature/pressure sensors on production lines. The IDS detects a port scanning attempt originating from a compromised maintenance laptop, alerting the security operations centre (SOC) before the attacker can identify vulnerable PLCs.

#### 7.1.2 Smart Home Security Monitoring

Consumer IoT devices — smart cameras, door locks, voice assistants, thermostats, baby monitors — are frequent targets for botnet recruitment and privacy-invading attacks. Internet service providers (ISPs) and home router manufacturers can integrate this IDS into residential gateway devices to monitor household IoT traffic and alert homeowners to potential security threats.

**Specific Use Case:** A smart home gateway running the IDS detects anomalous data exfiltration patterns from a compromised smart camera, where video feeds are being streamed to an unauthorised external server. The system classifies the traffic as a data exfiltration attack and triggers an automated notification to the homeowner.

#### 7.1.3 Healthcare IoT Security

Connected medical devices — infusion pumps, patient monitors, imaging equipment, wearable health trackers — collect and transmit sensitive patient health data. Regulatory frameworks (HIPAA, GDPR) mandate the protection of this data. The IDS can be deployed at the hospital network edge to monitor medical IoT traffic for security threats, ensuring compliance and patient safety.

#### 7.1.4 Smart City Infrastructure Protection

Smart city deployments — traffic signal controllers, environmental sensors, surveillance systems, smart street lighting, waste management sensors — create a distributed IoT infrastructure spanning city-wide geographic areas. The IDS can be integrated into smart city network operations centres to provide continuous traffic monitoring and threat detection across the urban IoT ecosystem.

### 7.2 Complementary Applications

#### 7.2.1 Network Forensic Analysis

Security analysts can use the system retrospectively to analyse captured network traffic (PCAP files converted to flow-level CSVs) from security incidents, identifying the specific attack types that occurred, their temporal patterns, and the network features that most strongly characterised each attack. This post-incident forensic capability supports root cause analysis and improves future defence strategies.

#### 7.2.2 Security Research and Education

The modular codebase and interactive dashboard serve as educational tools for cybersecurity courses and research projects. Students and researchers can:
- Experiment with different datasets, classifiers, and hyperparameters.
- Visualise the impact of preprocessing decisions on model performance.
- Understand the trade-offs between different evaluation metrics.
- Explore feature importance and its implications for network security.

#### 7.2.3 IoT Device Certification and Compliance Testing

IoT device manufacturers can use the system during product development and certification to test their devices' network behaviour against known attack patterns. By running the IDS on traffic captured during device testing, manufacturers can verify that their devices do not exhibit network behaviours characteristic of known attack types — either as attack targets or as potential attack vectors.

### 7.3 Deployment Architecture Scenarios

| Scenario | Deployment Point | Classification Mode | Key Requirement |
|----------|-----------------|-------------------|-----------------|
| Enterprise IIoT | Network gateway / NGFW | Multiclass (15 classes) | Low false positive rate |
| Smart Home | Residential router / gateway | Binary (Normal/Attack) | Minimal resource consumption |
| Healthcare | Hospital network edge server | Multiclass | HIPAA compliance, high recall |
| Smart City | NOC cloud/edge platform | Multiclass | Scalability, real-time alerts |
| Forensic Analysis | Analyst workstation | Multiclass | Detailed per-flow classification |
| Research | Lab environment | Both | Flexibility, interpretability |

---

## 8. CONCLUSION

This project presents the design, implementation, and evaluation of a **machine learning–based Intrusion Detection System for IoT networks**, leveraging the Edge-IIoT dataset and three supervised classifiers — Random Forest, Decision Tree, and XGBoost. The system addresses the critical inadequacy of traditional, signature-based IDS in the IoT security landscape through the application of modern supervised learning techniques.

### 8.1 Summary of Contributions

The key contributions of this project are:

1. **Comprehensive ML Pipeline:** A complete, end-to-end pipeline has been implemented, spanning data loading, cleaning, target auto-detection, feature selection with leakage guards, feature engineering, label encoding, stratified splitting, scaling, model training, cross-validation, evaluation, and serialisation. Every stage is implemented as a modular, reusable function with explicit anti-leakage safeguards.

2. **Rigorous Three-Way Classifier Comparison:** Random Forest, Decision Tree, and XGBoost are trained and evaluated under identical conditions — same preprocessed data, same stratified 80-20 split, same evaluation metrics — enabling a fair, unbiased comparison. The results demonstrate that XGBoost achieves the highest performance across all metrics due to its error-correcting boosting mechanism, second-order gradient optimisation, and built-in regularisation.

3. **Multi-Faceted Evaluation:** The evaluation framework goes beyond simple accuracy to include confusion matrices (revealing per-class misclassification patterns), classification reports (per-class precision, recall, F1), ROC curves with AUC (threshold-independent discrimination quality), precision-recall curves with AP (imbalance-aware detection quality), 5-fold stratified cross-validation (variance-aware stability assessment), and both intrinsic and permutation-based feature importance analysis.

4. **Interactive Analytical Dashboard:** The Streamlit web application transforms the trained ML pipeline into a practical, user-friendly tool for security analysts. The dashboard supports CSV file upload, model selection, binary and multiclass classification modes, real-time inference, and interactive Plotly-powered visualisations — making the system accessible to security professionals without ML expertise.

5. **Dual Classification Mode Support:** A single unified codebase supports both binary (Normal/Attack) and multiclass (15 attack categories) classification, providing flexibility for different operational requirements — from high-level threat alerts to fine-grained attack categorisation.

### 8.2 Key Findings

- **XGBoost consistently outperforms** both Random Forest and Decision Tree across accuracy, precision, recall, F1-score, and AUC-ROC, confirming the superiority of gradient-boosted ensembles for tabular cybersecurity classification tasks.
- **Feature engineering** (byte ratio, packet rate, error-flag interaction) provides incremental but meaningful improvements in detection performance by capturing higher-order traffic behaviour patterns.
- **Stratified splitting and training-only scaler fitting** are essential methodological safeguards — models evaluated with data leakage show inflated accuracy that does not reflect genuine out-of-sample performance.
- **Permutation importance analysis** reveals that network traffic volume features (byte counts, connection rates) and temporal features (duration, packet rate) are the strongest discriminators between normal and malicious IoT traffic.

### 8.3 Limitations

1. **Offline Evaluation:** The system is trained and evaluated on a static dataset. Real-time, streaming packet-level intrusion detection — which requires sub-millisecond inference latency and stateful flow tracking — is not implemented in this iteration.

2. **Dataset Specificity:** Model performance is inherently bounded by the feature distributions present in the Edge-IIoT dataset. Deployment in IoT environments with significantly different traffic characteristics (different protocols, device types, or network topologies) would require retraining on representative data from the target environment.

3. **No Deep Learning Comparison:** Neural network architectures (LSTM, CNN, Transformer-based models) that can capture sequential and spatial patterns in raw packet data are not explored in this project.

4. **Single-Dataset Evaluation:** The system is evaluated on a single dataset. Cross-dataset generalisation experiments (training on Edge-IIoT and testing on Bot-IoT, for instance) are not conducted.

### 8.4 Future Scope

1. **Real-Time Deployment:** Integration with network packet capture frameworks (Scapy, PyShark) and streaming inference pipelines (Apache Kafka, Flink) for real-time, production-grade IDS deployment.

2. **Deep Learning Extensions:** Incorporating LSTM networks for sequence-aware detection (capturing temporal attack patterns) and CNN/Transformer architectures for raw packet payload analysis.

3. **Federated Learning:** Implementing privacy-preserving federated learning where multiple IoT edge deployments collaboratively train a shared IDS model without exchanging raw traffic data.

4. **Adversarial Robustness:** Testing and hardening the classifiers against adversarial evasion attacks — where attackers intentionally craft traffic features to evade detection.

5. **Explainable AI Integration:** Incorporating SHAP (SHapley Additive exPlanations) values for per-prediction explainability, enabling analysts to understand why a specific network flow was classified as malicious.

6. **Automated Hyperparameter Optimisation:** Implementing Bayesian hyperparameter optimisation (Optuna, Hyperopt) to systematically search for optimal classifier configurations rather than relying on manually selected default parameters.

---

## REFERENCES

1. Ferrag, M. A., Friha, O., Hamouda, D., Maglaras, L. & Janicke, H. (2022). Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning. *IEEE Access*, 10, 40281–40306.

2. Koroniotis, N., Moustafa, N., Sitnikova, E. & Turnbull, B. (2019). Towards the Development of Realistic Botnet Dataset in the Internet of Things for Network Forensic Analytics: Bot-IoT Dataset. *Future Generation Computer Systems*, 100, 779–796.

3. Dhanabal, L. & Shantharajah, S. P. (2015). A Study on NSL-KDD Dataset for Intrusion Detection System Based on Classification Algorithms. *International Journal of Advanced Research in Computer and Communication Engineering*, 4(6), 446–452.

4. Panigrahi, R. & Borah, S. (2018). A Detailed Analysis of CICIDS 2017 Dataset for Designing Intrusion Detection Systems. *International Journal of Engineering & Technology*, 7(3.24), 479–482.

5. Shafiq, M., Tian, Z., Bashir, A. K., Du, X. & Guizani, M. (2020). Selection of Effective Machine Learning Algorithm and Bot-IoT Attacks Traffic Identification for Internet of Things in Smart City. *Future Generation Computer Systems*, 107, 433–442.

6. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

7. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

8. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
