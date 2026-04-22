
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
DATA_PATH   = "data/Data_for_UCI_named.csv"
OUTPUT_DIR  = "outputs/day2"
SPLITS_DIR  = "outputs/day1/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. LOAD CLEAN DATA
#    Re-run the cleaning steps from eda.py so this script is self-contained.
# =============================================================================
print("=" * 60)
print("STEP 1 — Load & clean dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

TARGET          = "stabf"
CONTINUOUS_TGT  = "stab"

# Median imputation
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# IQR capping on power coefficients and stab
iqr_cols = [c for c in df.columns if c.startswith("p")] + [CONTINUOUS_TGT]
for col in iqr_cols:
    Q1, Q3  = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR     = Q3 - Q1
    df[col] = df[col].clip(Q1 - 3 * IQR, Q3 + 3 * IQR)

# Drop leaking continuous target; encode binary target
df.drop(columns=[CONTINUOUS_TGT], inplace=True)
df[TARGET] = (df[TARGET] == "unstable").astype(int)

feature_cols = [c for c in df.columns if c != TARGET]
print(f"  Clean shape : {df.shape}  |  features : {feature_cols}")

# =============================================================================
# 2. TEMPORAL FEATURES (Rolling & Lag)
#    The dataset is a simulated time series (row order = time order).
#    We treat row index as the time axis.
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2 — Temporal Features (rolling + rate-of-change + lags)")
print("=" * 60)

df_feat = df.copy()

# ── 2a. Rolling statistics ──────────────────────────────────────────────────
# README proxies voltage via tau features and frequency via p features.
# We use all numeric features (excl. target) for rolling; README specifically
# calls out "voltage and frequency deviation" proxies.
# tau1-tau4  → reaction-time (voltage proxy)
# p1-p4      → power coefficients (frequency proxy)
rolling_features = [c for c in feature_cols if c.startswith("tau") or c.startswith("p")]
windows = [5, 10, 20]

for col in rolling_features:
    for w in windows:
        df_feat[f"roll_mean_{col}_w{w}"] = df_feat[col].rolling(w, min_periods=1).mean()
        df_feat[f"roll_std_{col}_w{w}"]  = df_feat[col].rolling(w, min_periods=1).std().fillna(0)
        df_feat[f"roll_min_{col}_w{w}"]  = df_feat[col].rolling(w, min_periods=1).min()
        df_feat[f"roll_max_{col}_w{w}"]  = df_feat[col].rolling(w, min_periods=1).max()

print(f"  Rolling stats added for {rolling_features} × windows {windows}")

# ── 2b. Rate-of-change (ΔV/Δt, ΔP/Δt) ─────────────────────────────────────
roc_features = [c for c in feature_cols if c.startswith("tau") or c.startswith("p")]
for col in roc_features:
    df_feat[f"roc_{col}"] = df_feat[col].diff().fillna(0)   # first-order finite diff

print(f"  Rate-of-change (diff) added for {roc_features}")

# ── 2c. Lag features (t-1, t-2, t-3) ───────────────────────────────────────
# README specifically mentions lagged stab; we also lag g (elasticity) features
# as they encode grid stress that propagates with delay.
lag_features = ["g1", "g2", "g3", "g4"] + [c for c in feature_cols if c.startswith("p")]
for col in lag_features:
    for lag in [1, 2, 3]:
        df_feat[f"lag{lag}_{col}"] = df_feat[col].shift(lag).bfill()

print(f"  Lag features (t-1,t-2,t-3) added for {lag_features}")

temporal_added = [c for c in df_feat.columns if c not in df.columns]
print(f"\n  Total temporal features added : {len(temporal_added)}")

# =============================================================================
# 3. GRAPH-DERIVED FEATURES
#    4-node star topology: node 0 = producer (tau1/p1/g1),
#                          nodes 1-3 = consumers (tau2-4 / p2-4 / g2-4)
#    Edges: producer↔each consumer (bidirectional).
#    Edge weight = edge_load_ratio = |power flow| / capacity proxy.
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3 — Graph-Derived Features (NetworkX)")
print("=" * 60)

def build_grid_graph(row: pd.Series) -> nx.Graph:
    """Build a weighted undirected graph for a single sample row."""
    G = nx.Graph()
    nodes = [0, 1, 2, 3]   # node 0 = producer, 1-3 = consumers
    G.add_nodes_from(nodes)

    # Capacity proxy: sum of |g| (price elasticity ~ line rating)
    # Load ratio proxy: |p_consumer| / capacity  (capped at 1)
    capacities = {
        (0, 1): abs(row["g1"]) + abs(row["g2"]) + 1e-9,
        (0, 2): abs(row["g1"]) + abs(row["g3"]) + 1e-9,
        (0, 3): abs(row["g1"]) + abs(row["g4"]) + 1e-9,
    }
    flows = {
        (0, 1): abs(row["p2"]),
        (0, 2): abs(row["p3"]),
        (0, 3): abs(row["p4"]),
    }
    for edge, cap in capacities.items():
        flow        = flows[edge]
        load_ratio  = min(flow / cap, 1.0)
        G.add_edge(*edge, weight=load_ratio, capacity=cap, flow=flow)

    return G

# We compute graph metrics per row — this is intentionally O(N·V) but the
# graph is tiny (4 nodes, 3 edges) so it remains fast.
print("  Computing per-row graph metrics (4-node star topology)…")

node_degree       = {i: [] for i in range(4)}
betweenness       = {i: [] for i in range(4)}
clustering        = {i: [] for i in range(4)}
edge_load_0_1_list = []
edge_load_0_2_list = []
edge_load_0_3_list = []
avg_edge_load_list = []

for _, row in df_feat.iterrows():
    G = build_grid_graph(row)

    deg   = dict(G.degree())
    bet   = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clust = nx.clustering(G, weight="weight")

    for n in range(4):
        node_degree[n].append(deg.get(n, 0))
        betweenness[n].append(bet.get(n, 0.0))
        clustering[n].append(clust.get(n, 0.0))

    el_01 = G[0][1]["weight"] if G.has_edge(0, 1) else 0.0
    el_02 = G[0][2]["weight"] if G.has_edge(0, 2) else 0.0
    el_03 = G[0][3]["weight"] if G.has_edge(0, 3) else 0.0
    edge_load_0_1_list.append(el_01)
    edge_load_0_2_list.append(el_02)
    edge_load_0_3_list.append(el_03)
    avg_edge_load_list.append(np.mean([el_01, el_02, el_03]))

# Attach graph features to dataframe
for n in range(4):
    df_feat[f"node{n}_degree"]      = node_degree[n]
    df_feat[f"node{n}_betweenness"] = betweenness[n]
    df_feat[f"node{n}_clustering"]  = clustering[n]

df_feat["edge_load_0_1"] = edge_load_0_1_list
df_feat["edge_load_0_2"] = edge_load_0_2_list
df_feat["edge_load_0_3"] = edge_load_0_3_list
df_feat["avg_edge_load"] = avg_edge_load_list

graph_features = (
    [f"node{n}_degree"      for n in range(4)] +
    [f"node{n}_betweenness" for n in range(4)] +
    [f"node{n}_clustering"  for n in range(4)] +
    ["edge_load_0_1", "edge_load_0_2", "edge_load_0_3", "avg_edge_load"]
)
print(f"  Graph features added : {graph_features}")

# =============================================================================
# 4. SUMMARY OF ENGINEERED FEATURE SET
# =============================================================================
all_feat_cols = [c for c in df_feat.columns if c != TARGET]
print("\n" + "=" * 60)
print("STEP 4 — Engineered Feature Set Summary")
print("=" * 60)
print(f"  Raw features              : {len(feature_cols)}")
print(f"  Temporal features added   : {len(temporal_added)}")
print(f"  Graph features added      : {len(graph_features)}")
print(f"  Total features            : {len(all_feat_cols)}")

# =============================================================================
# 5. TRAIN / VAL / TEST SPLIT ON ENGINEERED DATA
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5 — Split engineered dataset (70/15/15 stratified)")
print("=" * 60)

X = df_feat[all_feat_cols]
y = df_feat[TARGET]

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, stratify=y_trainval, random_state=42)

print(f"  Train : {len(X_train):>5}  |  class-1 rate: {y_train.mean():.3f}")
print(f"  Val   : {len(X_val):>5}  |  class-1 rate: {y_val.mean():.3f}")
print(f"  Test  : {len(X_test):>5}  |  class-1 rate: {y_test.mean():.3f}")

# =============================================================================
# 6. FEATURE SELECTION via Permutation Importance (Random Forest)
#    Retain features that contribute to cumulative 95% of importance.
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6 — Feature Selection (RF Permutation Importance → 95% cumulative)")
print("=" * 60)

# Scale for RF (tree models are scale-invariant, but keeps consistent with pipeline)
std_scaler  = StandardScaler()
X_train_std = pd.DataFrame(std_scaler.fit_transform(X_train), columns=X_train.columns)
X_val_std   = pd.DataFrame(std_scaler.transform(X_val),       columns=X_val.columns)
X_test_std  = pd.DataFrame(std_scaler.transform(X_test),      columns=X_test.columns)

# Fit a quick RF to rank features
rf_selector = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_selector.fit(X_train_std, y_train)

# Permutation importance on validation set
perm_imp = permutation_importance(
    rf_selector, X_val_std, y_val,
    n_repeats=10, random_state=42, n_jobs=-1
)
imp_mean = perm_imp.importances_mean
imp_df   = pd.DataFrame({
    "feature":    X_train.columns,
    "importance": imp_mean,
}).sort_values("importance", ascending=False).reset_index(drop=True)

# Clip negatives (irrelevant features) to 0 before cumulative sum
imp_df["importance"] = imp_df["importance"].clip(lower=0)
imp_df["cum_importance"] = imp_df["importance"].cumsum() / (imp_df["importance"].sum() + 1e-12)

# Select features up to 95% cumulative importance
selected_mask = imp_df["cum_importance"] <= 0.95
# Always keep at least one feature
if selected_mask.sum() == 0:
    selected_mask.iloc[0] = True
selected_features = imp_df.loc[selected_mask, "feature"].tolist()

# Safety: if 95% threshold is crossed at idx k, include feature k too
# (ensures we don't drop the feature that pushes us to/past 95%)
next_idx = selected_mask.sum()
if next_idx < len(imp_df):
    selected_features.append(imp_df.loc[next_idx, "feature"])

print(f"  Total features before selection : {len(all_feat_cols)}")
print(f"  Features retained (≥95% cum.)  : {len(selected_features)}")
print(f"\n  Top 20 features by permutation importance:")
print(imp_df.head(20).to_string(index=False))

dropped_features = [c for c in all_feat_cols if c not in selected_features]
print(f"\n  Dropped {len(dropped_features)} redundant features.")

# =============================================================================
# 7. VISUALISATIONS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7 — Visualisations")
print("=" * 60)

# ── 7a. Top-40 permutation importance bar chart ─────────────────────────────
top_n = min(40, len(imp_df))
fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
colors = ["#378ADD" if f in selected_features else "#BBBBBB" for f in imp_df["feature"][:top_n]]
ax.barh(imp_df["feature"][:top_n][::-1], imp_df["importance"][:top_n][::-1], color=colors[::-1])
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Mean permutation importance")
ax.set_title(f"Top {top_n} features — RF permutation importance\n(blue = selected, grey = dropped)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_permutation_importance.png", dpi=150)
plt.close()
print("  → Saved: 06_permutation_importance.png")

# ── 7b. Cumulative importance curve ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(imp_df) + 1), imp_df["cum_importance"], color="#378ADD", linewidth=2)
ax.axhline(0.95, color="#D85A30", linestyle="--", label="95% threshold")
ax.axvline(len(selected_features), color="#2CA02C", linestyle=":", label=f"Selected: {len(selected_features)} features")
ax.set_xlabel("Number of features (ranked by importance)")
ax.set_ylabel("Cumulative importance")
ax.set_title("Cumulative permutation importance — feature selection cutoff")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_cumulative_importance.png", dpi=150)
plt.close()
print("  → Saved: 07_cumulative_importance.png")

# ── 7c. Correlation heatmap — selected features (sample 30 for readability) ─
top_corr_features = selected_features[:30]
corr = df_feat[top_corr_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", center=0,
            linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation matrix — selected engineered features (top 30)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_engineered_correlation.png", dpi=150)
plt.close()
print("  → Saved: 08_engineered_correlation.png")

# ── 7d. Distribution of key graph features by class ─────────────────────────
graph_plot_cols = ["avg_edge_load", "node0_betweenness", "node1_clustering", "edge_load_0_1"]
fig, axes = plt.subplots(1, len(graph_plot_cols), figsize=(16, 4))
for ax, col in zip(axes, graph_plot_cols):
    df_feat[df_feat[TARGET] == 0][col].plot.hist(ax=ax, bins=40, alpha=0.6, color="#378ADD", label="stable")
    df_feat[df_feat[TARGET] == 1][col].plot.hist(ax=ax, bins=40, alpha=0.6, color="#D85A30", label="unstable")
    ax.set_title(col, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=9)
fig.suptitle("Graph-derived feature distributions by class", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_graph_features_by_class.png", dpi=150)
plt.close()
print("  → Saved: 09_graph_features_by_class.png")

# ── 7e. Sample rolling feature vs raw — show temporal enrichment ─────────────
sample_col = "p1"
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
sample_slice = df_feat.iloc[:200]
axes[0].plot(sample_slice[sample_col].values, color="#378ADD", linewidth=1, label=f"raw {sample_col}")
axes[0].set_ylabel("Raw value"); axes[0].legend(fontsize=9)
axes[1].plot(sample_slice[f"roll_mean_{sample_col}_w10"].values, color="#2CA02C", linewidth=1, label="rolling mean w=10")
axes[1].plot(sample_slice[f"roll_std_{sample_col}_w10"].values,  color="#D85A30", linewidth=1, label="rolling std w=10")
axes[1].set_ylabel("Rolling stats"); axes[1].legend(fontsize=9)
axes[2].plot(sample_slice[f"roc_{sample_col}"].values, color="#9467BD", linewidth=1, label="rate of change (diff)")
axes[2].set_ylabel("ΔP/Δt"); axes[2].legend(fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
axes[2].set_xlabel("Sample index")
fig.suptitle(f"Temporal feature enrichment — {sample_col} (first 200 rows)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_temporal_enrichment.png", dpi=150)
plt.close()
print("  → Saved: 10_temporal_enrichment.png")

# =============================================================================
# 8. APPLY SELECTION & SAVE FINAL SPLITS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8 — Save selected feature splits")
print("=" * 60)

splits_dir = f"{OUTPUT_DIR}/splits"
os.makedirs(splits_dir, exist_ok=True)

# Rebuild splits with ONLY selected features
X_train_sel = X_train[selected_features]
X_val_sel   = X_val[selected_features]
X_test_sel  = X_test[selected_features]

# Standard scaled
std_sel = StandardScaler()
X_train_sel_std = pd.DataFrame(std_sel.fit_transform(X_train_sel), columns=selected_features)
X_val_sel_std   = pd.DataFrame(std_sel.transform(X_val_sel),       columns=selected_features)
X_test_sel_std  = pd.DataFrame(std_sel.transform(X_test_sel),      columns=selected_features)

# MinMax scaled
mm_sel = MinMaxScaler()
X_train_sel_mm  = pd.DataFrame(mm_sel.fit_transform(X_train_sel),  columns=selected_features)
X_val_sel_mm    = pd.DataFrame(mm_sel.transform(X_val_sel),        columns=selected_features)
X_test_sel_mm   = pd.DataFrame(mm_sel.transform(X_test_sel),       columns=selected_features)

# SMOTE on std-scaled train
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_sel_std, y_train)

# Save raw selected
X_train_sel.to_csv(f"{splits_dir}/X_train_eng_raw.csv", index=False)
X_val_sel.to_csv(f"{splits_dir}/X_val_eng_raw.csv",     index=False)
X_test_sel.to_csv(f"{splits_dir}/X_test_eng_raw.csv",   index=False)

# Save std scaled selected
X_train_sel_std.to_csv(f"{splits_dir}/X_train_eng_std.csv", index=False)
X_val_sel_std.to_csv(f"{splits_dir}/X_val_eng_std.csv",     index=False)
X_test_sel_std.to_csv(f"{splits_dir}/X_test_eng_std.csv",   index=False)

# Save mm scaled selected
X_train_sel_mm.to_csv(f"{splits_dir}/X_train_eng_mm.csv",   index=False)
X_val_sel_mm.to_csv(f"{splits_dir}/X_val_eng_mm.csv",       index=False)
X_test_sel_mm.to_csv(f"{splits_dir}/X_test_eng_mm.csv",     index=False)

# Save SMOTE
pd.DataFrame(X_train_smote, columns=selected_features).to_csv(
    f"{splits_dir}/X_train_eng_smote.csv", index=False)

# Labels
y_train.to_csv(f"{splits_dir}/y_train.csv", index=False)
y_val.to_csv(f"{splits_dir}/y_val.csv",     index=False)
y_test.to_csv(f"{splits_dir}/y_test.csv",   index=False)
pd.Series(y_train_smote, name=TARGET).to_csv(f"{splits_dir}/y_train_smote.csv", index=False)

# Save the selected feature list for downstream scripts
pd.Series(selected_features, name="feature").to_csv(
    f"{splits_dir}/selected_features.csv", index=False)

# Save the full importance table
imp_df.to_csv(f"{OUTPUT_DIR}/feature_importance_table.csv", index=False)

print(f"  Saved to : {splits_dir}/")
print("  Files    : X_train/val/test_eng_raw, _std, _mm, _smote")
print("             y_train, y_val, y_test, y_train_smote")
print("             selected_features.csv")
print(f"  Importance table → {OUTPUT_DIR}/feature_importance_table.csv")

# =============================================================================
# 9. SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("Feature Engineering Summary")
print("=" * 60)
print(f"  Original features         : {len(feature_cols)}")
print(f"  + Temporal features       : {len(temporal_added)}")
print(f"  + Graph features          : {len(graph_features)}")
print(f"  = Total engineered        : {len(all_feat_cols)}")
print(f"  → Selected (95% cum imp)  : {len(selected_features)}")
print(f"  Dropped (redundant)       : {len(dropped_features)}")
print(f"  SMOTE train size          : {len(X_train_smote)}")
print(f"  Plots saved to            : {OUTPUT_DIR}/")
print(f"  Splits saved to           : {splits_dir}/")
