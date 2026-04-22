
 
import os
import urllib.request   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")
 
## ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/Data_for_UCI_named.csv"
OUTPUT_DIR  = "outputs/day1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Auto-download if dataset not present
import urllib.request

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv"

if not os.path.exists(DATA_PATH):
    print("Dataset not found. Downloading...")
    urllib.request.urlretrieve(url, DATA_PATH)
    print("Download complete!")
else:
    print("Dataset already exists.")

# =============================================================================
# 1. LOAD & INITIAL INSPECTION
# =============================================================================
print("=" * 60)
print("STEP 1 — Loading dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
 
print(f"\nShape         : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumns       :\n{list(df.columns)}")
print(f"\nData types    :\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nFirst 5 rows  :\n{df.head()}")

TARGET = "stabf"
CONTINUOUS_TARGET = "stab"  
 
print(f"\nTarget distribution ('{TARGET}'):")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True).round(3))
 
# =============================================================================
# 2. EDA PLOTS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2 — Exploratory Data Analysis")
print("=" * 60)
 
feature_cols = [c for c in df.columns if c not in [TARGET, CONTINUOUS_TARGET]]
 
# --- 2a. Class distribution bar chart ---
fig, ax = plt.subplots(figsize=(5, 3))
counts = df[TARGET].value_counts()
bars = ax.bar(counts.index, counts.values, color=["#378ADD", "#D85A30"], width=0.5)
ax.bar_label(bars, fmt="%d", padding=4, fontsize=11)
ax.set_title("Class distribution", fontsize=13)
ax.set_ylabel("Count")
ax.set_xlabel("stabf")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_class_distribution.png", dpi=150)
plt.close()
print("  → Saved: 01_class_distribution.png")
 
# --- 2b. Feature distributions (histogram grid) ---
n = len(feature_cols)
ncols = 4
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 2.8))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].hist(df[df[TARGET] == "stable"][col],   bins=40, alpha=0.6, label="stable",   color="#378ADD")
    axes[i].hist(df[df[TARGET] == "unstable"][col], bins=40, alpha=0.6, label="unstable", color="#D85A30")
    axes[i].set_title(col, fontsize=10)
    axes[i].spines[["top", "right"]].set_visible(False)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
axes[0].legend(fontsize=9)
fig.suptitle("Feature distributions by class", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved: 02_feature_distributions.png")
 
# --- 2c. Correlation heatmap (numeric features only) ---
numeric_df = df[feature_cols + [CONTINUOUS_TARGET]].select_dtypes(include=np.number)
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation matrix — raw features", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_correlation_heatmap.png", dpi=150)
plt.close()
print("  → Saved: 03_correlation_heatmap.png")
 
# --- 2d. Boxplots: top features by class ---
top_features = feature_cols[:8]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(top_features):
    df.boxplot(column=col, by=TARGET, ax=axes[i], grid=False,
               boxprops=dict(color="#378ADD"),
               medianprops=dict(color="#D85A30", linewidth=2))
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].spines[["top", "right"]].set_visible(False)
plt.suptitle("Feature boxplots by class", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_boxplots_by_class.png", dpi=150)
plt.close()
print("  → Saved: 04_boxplots_by_class.png")
 
# =============================================================================
# 3. DATA CLEANING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3 — Data cleaning")
print("=" * 60)
 
df_clean = df.copy()
 
# 3a. Missing value imputation
for col in df_clean.select_dtypes(include=np.number).columns:
    if df_clean[col].isnull().any():
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
        print(f"  Imputed (median): {col}")
 
# 3b. IQR outlier filtering on key numeric features
#     (power coefficients p1-p4 and frequency deviation stab)
iqr_cols = [c for c in feature_cols if c.startswith("p")] + [CONTINUOUS_TARGET]
before = len(df_clean)
for col in iqr_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR   # 3×IQR — conservative to preserve rare fault events
    outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    if outliers:
        print(f"  IQR filter on '{col}': {outliers} outliers capped (not dropped)")
        df_clean[col] = df_clean[col].clip(lower, upper)
 
after = len(df_clean)
print(f"\n  Rows before cleaning : {before}")
print(f"  Rows after  cleaning : {after}  (outliers capped, not removed)")
 
# 3c. Drop continuous stability column (stab) — use stabf as target
#     stab is the underlying float from which stabf is derived; keeping it would leak the target
df_clean.drop(columns=[CONTINUOUS_TARGET], inplace=True)
print(f"\n  Dropped '{CONTINUOUS_TARGET}' (target-leaking continuous version)")
 
# 3d. Encode target: stable=0, unstable=1
df_clean[TARGET] = (df_clean[TARGET] == "unstable").astype(int)
print(f"  Encoded target: stable=0, unstable=1")
print(f"\n  Clean dataset shape: {df_clean.shape}")
 
# =============================================================================
# 4. TRAIN / VAL / TEST SPLIT
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4 — Train / Val / Test split (70 / 15 / 15)")
print("=" * 60)
 
X = df_clean.drop(columns=[TARGET])
y = df_clean[TARGET]
 
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, stratify=y_trainval, random_state=42)
# 0.1765 of 0.85 ≈ 0.15 of total
 
print(f"  Train : {len(X_train):>5} samples  |  class 1 rate: {y_train.mean():.3f}")
print(f"  Val   : {len(X_val):>5} samples  |  class 1 rate: {y_val.mean():.3f}")
print(f"  Test  : {len(X_test):>5} samples  |  class 1 rate: {y_test.mean():.3f}")
 
# =============================================================================
# 5. SCALING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5 — Feature scaling")
print("=" * 60)
 
# StandardScaler — for RF, XGBoost (tree models are scale-invariant but this
# normalizes for PCA/IF downstream)
std_scaler = StandardScaler()
X_train_std = pd.DataFrame(std_scaler.fit_transform(X_train), columns=X_train.columns)
X_val_std   = pd.DataFrame(std_scaler.transform(X_val),       columns=X_val.columns)
X_test_std  = pd.DataFrame(std_scaler.transform(X_test),      columns=X_test.columns)
 
# MinMaxScaler — bounded [0,1] inputs for Isolation Forest and PCA
mm_scaler = MinMaxScaler()
X_train_mm = pd.DataFrame(mm_scaler.fit_transform(X_train), columns=X_train.columns)
X_val_mm   = pd.DataFrame(mm_scaler.transform(X_val),       columns=X_val.columns)
X_test_mm  = pd.DataFrame(mm_scaler.transform(X_test),      columns=X_test.columns)
 
print("  StandardScaler  → X_train_std, X_val_std, X_test_std")
print("  MinMaxScaler    → X_train_mm,  X_val_mm,  X_test_mm")
print(f"\n  Sample means after StandardScaler (should be ~0):")
print(X_train_std.mean().round(4).to_string())
 
# =============================================================================
# 6. CLASS IMBALANCE — SMOTE balancing
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6 — Class imbalance handling")
print("=" * 60)
 
# Apply SMOTE on TRAIN split ONLY
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_std, y_train)
 
print(f"\n  Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"  After  SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")
 
# Visualize SMOTE effect
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(["stable (0)", "unstable (1)"], y_train.value_counts().sort_index().values,
            color=["#378ADD", "#D85A30"])
axes[0].set_title("Before SMOTE")
axes[0].set_ylabel("Count")
axes[1].bar(["stable (0)", "unstable (1)"], pd.Series(y_train_smote).value_counts().sort_index().values,
            color=["#378ADD", "#D85A30"])
axes[1].set_title("After SMOTE (train only)")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.suptitle("Class balance: training set", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_smote_balance.png", dpi=150)
plt.close()
print("  → Saved: 05_smote_balance.png")
 
# =============================================================================
# 7. SAVE ALL SPLITS TO DISK
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7 — Saving splits to disk")
print("=" * 60)
 
splits_dir = f"{OUTPUT_DIR}/splits"
os.makedirs(splits_dir, exist_ok=True)
 
# Raw splits (unscaled) — for tree models that don't need scaling
X_train.to_csv(f"{splits_dir}/X_train_raw.csv",    index=False)
X_val.to_csv(f"{splits_dir}/X_val_raw.csv",        index=False)
X_test.to_csv(f"{splits_dir}/X_test_raw.csv",      index=False)
 
# StandardScaled splits — for RF, XGBoost, ensemble
X_train_std.to_csv(f"{splits_dir}/X_train_std.csv",  index=False)
X_val_std.to_csv(f"{splits_dir}/X_val_std.csv",      index=False)
X_test_std.to_csv(f"{splits_dir}/X_test_std.csv",    index=False)
 
# MinMaxScaled splits — for Isolation Forest, PCA
X_train_mm.to_csv(f"{splits_dir}/X_train_mm.csv",  index=False)
X_val_mm.to_csv(f"{splits_dir}/X_val_mm.csv",      index=False)
X_test_mm.to_csv(f"{splits_dir}/X_test_mm.csv",    index=False)
 
# SMOTE-balanced train split (std-scaled)
pd.DataFrame(X_train_smote, columns=X_train.columns).to_csv(
    f"{splits_dir}/X_train_smote.csv", index=False)
 
# Labels
y_train.to_csv(f"{splits_dir}/y_train.csv",               index=False)
y_val.to_csv(f"{splits_dir}/y_val.csv",                   index=False)
y_test.to_csv(f"{splits_dir}/y_test.csv",                 index=False)
pd.Series(y_train_smote, name="stabf").to_csv(
    f"{splits_dir}/y_train_smote.csv", index=False)
 
print(f"  Saved to: {splits_dir}/")
print("  Files: X_train_raw, X_val_raw, X_test_raw")
print("         X_train_std, X_val_std, X_test_std")
print("         X_train_mm,  X_val_mm,  X_test_mm")
print("         X_train_smote, y_train, y_val, y_test, y_train_smote")
 
# =============================================================================
# 8. EDA AND PREPROCESSING SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("DAY 1 COMPLETE — Summary")
print("=" * 60)
print(f"  Dataset       : {df.shape[0]} samples, {len(feature_cols)} raw features")
print(f"  After cleaning: {df_clean.shape[0]} samples (outliers capped, no rows dropped)")
print(f"  Train / Val / Test split: {len(X_train)} / {len(X_val)} / {len(X_test)}")
print(f"  SMOTE train size         : {len(X_train_smote)}")
print(f"  Scalers ready            : StandardScaler + MinMaxScaler (fit on train only)")
print(f"  EDA plots saved to       : {OUTPUT_DIR}/")
print(f"  Splits saved to          : {splits_dir}/")

 
