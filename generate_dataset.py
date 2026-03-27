"""
generate_dataset.py — Synthetic IoT Network Traffic Dataset Generator
=====================================================================
Generates a realistic synthetic dataset of IoT network flows with:
  - 14 base features + 4 engineered features mimicking real traffic
  - Realistic class imbalance (Normal traffic dominates)
  - Overlapping distributions with injected noise for realism
  - Binary labels  (0 = Normal, 1 = Attack)
  - Multiclass labels (0 = Normal, 1 = DoS, 2 = Port Scan, 3 = Data Exfiltration)

Usage
-----
    python generate_dataset.py
"""

import os
import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────
TOTAL_SAMPLES = 10_000
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.csv")
SEED = 42

# Realistic class distribution — normal traffic dominates in real
# IoT networks (Meidan et al., 2018; Koroniotis et al., 2019).
CLASS_RATIOS = {
    "Normal":            0.50,   # 50 %
    "DoS":               0.20,   # 20 %
    "Port Scan":         0.15,   # 15 %
    "Data Exfiltration": 0.15,   # 15 %
}

# Noise magnitude — fraction of the feature scale added as Gaussian
# noise to simulate sensor jitter and measurement uncertainty.
NOISE_FRACTION = 0.08


# ── Noise injection helper ────────────────────────────────────────

def _add_noise(arr: np.ndarray, rng: np.random.Generator,
               fraction: float = NOISE_FRACTION) -> np.ndarray:
    """Add Gaussian noise proportional to the array's std deviation."""
    noise = rng.normal(0, fraction * (arr.std() + 1e-8), size=arr.shape)
    return arr + noise


# ── Per-class feature generation ──────────────────────────────────

def _generate_normal(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Normal IoT traffic: moderate packet sizes, regular durations,
    low error rates, balanced byte counts.
    """
    data = pd.DataFrame({
        "packet_size":     rng.normal(loc=250, scale=80, size=n).clip(64, 1500),
        "duration":        rng.exponential(scale=15, size=n).clip(0.1, 120),
        "src_bytes":       rng.normal(loc=500, scale=200, size=n).clip(0),
        "dst_bytes":       rng.normal(loc=450, scale=180, size=n).clip(0),
        "wrong_fragment":  rng.poisson(lam=0.05, size=n),
        "urgent":          rng.poisson(lam=0.01, size=n),
        "count":           rng.poisson(lam=5, size=n),
        "srv_count":       rng.poisson(lam=3, size=n),
        "protocol_type":   rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2]),
        "connection_rate": rng.normal(loc=10, scale=4, size=n).clip(0),
        "error_rate":      rng.beta(a=1, b=20, size=n),
        "flag":            rng.choice([0, 1, 2, 3], size=n, p=[0.6, 0.2, 0.1, 0.1]),
        "land":            rng.choice([0, 1], size=n, p=[0.98, 0.02]),
        "logged_in":       rng.choice([0, 1], size=n, p=[0.3, 0.7]),
    })
    # Inject noise on continuous features for realism
    for col in ["packet_size", "duration", "src_bytes", "dst_bytes",
                "connection_rate", "error_rate"]:
        data[col] = _add_noise(data[col].values, rng)
    return data


def _generate_dos(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    DoS attack: very HIGH packet sizes, extremely HIGH connection rate,
    elevated error rate, many connections in bursts.
    """
    data = pd.DataFrame({
        "packet_size":     rng.normal(loc=1200, scale=200, size=n).clip(800, 1500),
        "duration":        rng.exponential(scale=2, size=n).clip(0.01, 10),
        "src_bytes":       rng.normal(loc=2000, scale=600, size=n).clip(500),
        "dst_bytes":       rng.normal(loc=200, scale=100, size=n).clip(0),
        "wrong_fragment":  rng.poisson(lam=1.5, size=n),
        "urgent":          rng.poisson(lam=0.3, size=n),
        "count":           rng.poisson(lam=80, size=n),
        "srv_count":       rng.poisson(lam=10, size=n),
        "protocol_type":   rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]),
        "connection_rate": rng.normal(loc=85, scale=15, size=n).clip(50),
        "error_rate":      rng.beta(a=3, b=5, size=n),
        "flag":            rng.choice([0, 1, 2, 3], size=n, p=[0.2, 0.5, 0.2, 0.1]),
        "land":            rng.choice([0, 1], size=n, p=[0.85, 0.15]),
        "logged_in":       rng.choice([0, 1], size=n, p=[0.8, 0.2]),
    })
    for col in ["packet_size", "duration", "src_bytes", "dst_bytes",
                "connection_rate", "error_rate"]:
        data[col] = _add_noise(data[col].values, rng)
    return data


def _generate_port_scan(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Port Scan: very SHORT durations, HIGH srv_count (many distinct
    services probed), small packets, low data transfer.
    """
    data = pd.DataFrame({
        "packet_size":     rng.normal(loc=120, scale=30, size=n).clip(40, 300),
        "duration":        rng.exponential(scale=0.5, size=n).clip(0.001, 2),
        "src_bytes":       rng.normal(loc=100, scale=50, size=n).clip(0),
        "dst_bytes":       rng.normal(loc=80, scale=40, size=n).clip(0),
        "wrong_fragment":  rng.poisson(lam=0.1, size=n),
        "urgent":          rng.poisson(lam=0.02, size=n),
        "count":           rng.poisson(lam=50, size=n),
        "srv_count":       rng.poisson(lam=45, size=n),
        "protocol_type":   rng.choice([0, 1, 2], size=n, p=[0.6, 0.25, 0.15]),
        "connection_rate": rng.normal(loc=40, scale=10, size=n).clip(15),
        "error_rate":      rng.beta(a=2, b=8, size=n),
        "flag":            rng.choice([0, 1, 2, 3], size=n, p=[0.3, 0.3, 0.3, 0.1]),
        "land":            rng.choice([0, 1], size=n, p=[0.95, 0.05]),
        "logged_in":       rng.choice([0, 1], size=n, p=[0.9, 0.1]),
    })
    for col in ["packet_size", "duration", "src_bytes", "dst_bytes",
                "connection_rate", "error_rate"]:
        data[col] = _add_noise(data[col].values, rng)
    return data


def _generate_exfiltration(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Data Exfiltration: very HIGH dst_bytes (large outbound data),
    longer durations, moderate connection rates.
    """
    data = pd.DataFrame({
        "packet_size":     rng.normal(loc=900, scale=150, size=n).clip(500, 1500),
        "duration":        rng.normal(loc=60, scale=25, size=n).clip(10, 200),
        "src_bytes":       rng.normal(loc=300, scale=100, size=n).clip(0),
        "dst_bytes":       rng.normal(loc=5000, scale=1500, size=n).clip(2000),
        "wrong_fragment":  rng.poisson(lam=0.2, size=n),
        "urgent":          rng.poisson(lam=0.05, size=n),
        "count":           rng.poisson(lam=8, size=n),
        "srv_count":       rng.poisson(lam=2, size=n),
        "protocol_type":   rng.choice([0, 1, 2], size=n, p=[0.4, 0.35, 0.25]),
        "connection_rate": rng.normal(loc=15, scale=5, size=n).clip(2),
        "error_rate":      rng.beta(a=1, b=15, size=n),
        "flag":            rng.choice([0, 1, 2, 3], size=n, p=[0.4, 0.2, 0.2, 0.2]),
        "land":            rng.choice([0, 1], size=n, p=[0.92, 0.08]),
        "logged_in":       rng.choice([0, 1], size=n, p=[0.4, 0.6]),
    })
    for col in ["packet_size", "duration", "src_bytes", "dst_bytes",
                "connection_rate", "error_rate"]:
        data[col] = _add_noise(data[col].values, rng)
    return data


# ── Class Imbalance Analysis ─────────────────────────────────────

def print_class_imbalance_analysis(df: pd.DataFrame) -> None:
    """Print a detailed class imbalance analysis with ratios and warnings."""
    print("\n" + "─" * 60)
    print("  CLASS IMBALANCE ANALYSIS")
    print("─" * 60)

    label_counts = df["label"].value_counts()
    total = len(df)
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count

    print(f"\n  Total samples     : {total:,}")
    print(f"  Number of classes : {label_counts.shape[0]}")
    print(f"  Imbalance ratio   : {imbalance_ratio:.2f}:1 "
          f"(majority / minority)\n")

    print(f"  {'Class':25s} {'Count':>7s} {'Percent':>8s} {'Ratio':>8s}")
    print(f"  {'─' * 50}")
    for cls, cnt in label_counts.items():
        pct = cnt / total * 100
        ratio = cnt / min_count
        print(f"  {cls:25s} {cnt:7,d} {pct:7.1f}% {ratio:7.2f}x")

    if imbalance_ratio > 3.0:
        print(f"\n  ⚠️  WARNING: Significant class imbalance detected "
              f"(ratio {imbalance_ratio:.1f}:1).")
        print("     Consider using class_weight='balanced' or SMOTE.")
    elif imbalance_ratio > 1.5:
        print(f"\n  ℹ️  Moderate imbalance (ratio {imbalance_ratio:.1f}:1) — "
              "handled by class_weight='balanced'.")
    else:
        print(f"\n  ✅ Classes are reasonably balanced.")

    print("─" * 60)


# ── Main generator ────────────────────────────────────────────────

def generate_dataset(
    total: int = TOTAL_SAMPLES,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Build the full synthetic dataset by combining per-class generators.

    Returns a shuffled DataFrame with columns:
        [features…, label, attack_type]
    """
    rng = np.random.default_rng(seed)

    # Compute per-class sample counts
    counts = {}
    remaining = total
    for i, (cls, ratio) in enumerate(CLASS_RATIOS.items()):
        if i == len(CLASS_RATIOS) - 1:
            counts[cls] = remaining          # last class gets the rest
        else:
            n = int(round(total * ratio))
            counts[cls] = n
            remaining -= n

    # Generate each class
    generators = {
        "Normal":            _generate_normal,
        "DoS":               _generate_dos,
        "Port Scan":         _generate_port_scan,
        "Data Exfiltration": _generate_exfiltration,
    }

    multiclass_map = {
        "Normal":            0,
        "DoS":               1,
        "Port Scan":         2,
        "Data Exfiltration": 3,
    }

    frames = []
    for cls, n in counts.items():
        df_cls = generators[cls](n, rng)
        df_cls["label"] = cls                         # human-readable label
        df_cls["attack_type"] = multiclass_map[cls]   # numeric multiclass
        frames.append(df_cls)

    df = pd.concat(frames, ignore_index=True)

    # Round numeric columns for cleanliness
    float_cols = df.select_dtypes(include=[np.floating]).columns
    df[float_cols] = df[float_cols].round(4)

    # Shuffle rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  IoT Synthetic Dataset Generator")
    print("=" * 60)

    df = generate_dataset()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    # Report
    print(f"\n  Dataset shape : {df.shape}")
    print(f"  Saved to      : {os.path.abspath(OUTPUT_FILE)}")

    # Class distribution
    print("\n  ── Class distribution (label) ──")
    label_counts = df["label"].value_counts()
    for cls, cnt in label_counts.items():
        pct = cnt / len(df) * 100
        print(f"    {cls:25s} : {cnt:5d}  ({pct:.1f}%)")

    print("\n  ── Multiclass distribution (attack_type) ──")
    at_counts = df["attack_type"].value_counts().sort_index()
    names = {0: "Normal", 1: "DoS", 2: "Port Scan", 3: "Data Exfiltration"}
    for code, cnt in at_counts.items():
        pct = cnt / len(df) * 100
        print(f"    {code} = {names[code]:25s} : {cnt:5d}  ({pct:.1f}%)")

    # Class imbalance analysis
    print_class_imbalance_analysis(df)

    # Feature statistics
    print("\n  ── Feature statistics (numeric columns) ──")
    feature_cols = [c for c in df.columns if c not in ("label", "attack_type")]
    print(df[feature_cols].describe().round(2).to_string())

    print(f"\n  ── Feature sample (first 5 rows) ──")
    print(df.head().to_string(index=False))
    print("\n✅  Dataset generation complete!\n")
