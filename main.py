"""
IF2123 Linear Algebra & Geometry
Paper: Implementation of a System for Clustering and Stability Analysis of
Price Dynamics of Traditional Market Commodities Based on Vector Spaces and Eigenvalues

DATA:
- Using the provided 'data.csv' file.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Libraries for vector & matrix analysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# 1) PREPROCESSING & FEATURE UTILITIES (VECTOR SPACE)
# =========================

def compute_log_returns(price_series: pd.Series) -> pd.Series:
    """
    Calculate log-returns: r_t = ln(p_t) - ln(p_{t-1}).
    This transformation normalizes price scales so that
    expensive and inexpensive commodities can be compared fairly.
    """
    p = price_series.astype(float)
    
    # Replace zero values with NaN, then forward-fill to avoid log(0)
    p = p.replace(0, np.nan).ffill().fillna(1.0)
    return np.log(p).diff().fillna(0.0)

def extract_features_from_series(r: pd.Series) -> dict:
    """
    Construct a commodity feature vector x_i ∈ R^d from the return series r(t).
    These features represent the price 'behavior' of a commodity in vector space.
    """
    y = r.to_numpy(dtype=float)
    T = len(y)

    # Descriptive statistics as vector components
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=1)) if T > 1 else 0.0
    var = std ** 2
    
    # Simple linear trend (slope)
    t_arr = np.arange(T, dtype=float)
    if T >= 2:
        cov_ty = float(np.cov(t_arr, y, ddof=1)[0, 1])
        var_t = float(np.var(t_arr, ddof=1))
        slope = cov_ty / var_t if var_t > 0 else 0.0
    else:
        slope = 0.0

    # Lag-1 autocorrelation (short-term memory)
    if T >= 3 and np.std(y[:-1]) > 1e-12 and np.std(y[1:]) > 1e-12:
        acf1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    else:
        acf1 = 0.0

    return {
        "mean": mean,
        "std": std,
        "var": var,
        "slope": slope,
        "acf1": acf1
    }

# =========================
# 2) CLUSTERING (K-MEANS ALGORITHM)
# =========================

def choose_k_by_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 8, seed: int = 2123) -> int:
    """
    Select the optimal number of clusters (k) using the Silhouette Score.
    """
    best_k = k_min
    best_score = -1.0

    # Limit k_max so it does not exceed the number of available samples
    max_possible = min(k_max, X.shape[0] - 1)
    if max_possible < k_min:
        return k_min

    for k in range(k_min, max_possible + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        
        # Skip trivial clustering (single cluster)
        if len(set(labels)) < 2:
            continue
            
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k

# =========================
# 3) DYNAMICAL MODEL & EIGENVALUES (STABILITY)
# =========================

def build_cluster_state_series(returns_wide: pd.DataFrame, members: list[str]) -> np.ndarray:
    """
    Construct the state matrix S(t) for a given cluster.
    The state consists of:
    - Mean return of the cluster
    - Standard deviation of returns within the cluster
    """
    sub = returns_wide[members].copy().fillna(0.0)
    
    # State 1: Average price movement of the cluster
    s1 = sub.mean(axis=1)
    
    # State 2: Dispersion/volatility within the cluster
    s2 = sub.std(axis=1, ddof=1).fillna(0.0)
    
    # Combine into a T x 2 state matrix
    S = pd.DataFrame({"mean": s1, "std": s2}, index=sub.index)
    return S.to_numpy()

def estimate_var1_matrix(S: np.ndarray, ridge_eps: float = 1e-6) -> np.ndarray:
    """
    Estimate the transition matrix A using Least Squares:
    s(t+1) = A * s(t)
    """
    T, m = S.shape
    if T < 3:
        return np.eye(m)  # Fallback if historical data are insufficient

    Z = S[:-1].T  # Current state matrix (input)
    Y = S[1:].T   # Next-day state matrix (target)

    # A = Y * Z^T * inv(Z * Z^T + regularization)
    ZZt = Z @ Z.T
    A = (Y @ Z.T) @ np.linalg.inv(ZZt + ridge_eps * np.eye(m))
    return A

def spectral_radius_and_eigs(A: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute the spectral radius (maximum absolute eigenvalue).
    rho < 1 : Stable system
    rho > 1 : Unstable system
    """
    eigvals = np.linalg.eigvals(A)
    rho = float(np.max(np.abs(eigvals)))
    return rho, eigvals

@dataclass
class ClusterStabilityResult:
    cluster_id: int
    n_members: int
    members: list[str]
    A: np.ndarray
    eigvals: np.ndarray
    spectral_radius: float
    stable: bool

# =========================
# 4) VISUALIZATION MODULE
# =========================

def visualize_results(X: np.ndarray, labels: np.ndarray, 
                      feat_df: pd.DataFrame, stability_results: list[ClusterStabilityResult]):
    """
    Integrated visualization function for generating publication-ready figures.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- PLOT 1: Feature Space Map (PCA Projection) ---
    # Project d-dimensional features into 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Clustering Scatter Plot
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap='viridis',
        s=100, alpha=0.8, edgecolors='k'
    )
    
    # Annotate selected points (commodity names)
    for i, txt in enumerate(feat_df.index):
        if i % 2 == 0 or abs(X_pca[i, 0]) > 2 or abs(X_pca[i, 1]) > 2:
            plt.annotate(
                txt,
                (X_pca[i, 0] + 0.05, X_pca[i, 1] + 0.05),
                fontsize=8, alpha=0.9
            )
            
    plt.title('Commodity Clustering Map (2D PCA Projection)', fontsize=12, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} Variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} Variance)')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Spectral Radius (Stability Analysis)
    plt.subplot(1, 2, 2)
    
    cluster_ids = [res.cluster_id for res in stability_results]
    radii = [res.spectral_radius for res in stability_results]
    colors = ['green' if r < 1.0 else 'red' for r in radii]
    
    bars = plt.bar(cluster_ids, radii, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Stability Threshold (ρ = 1)')
    
    plt.title('Dynamical Stability Analysis per Cluster', fontsize=12, fontweight='bold')
    plt.xlabel('Cluster ID')
    plt.ylabel('Spectral Radius (ρ)')
    plt.xticks(cluster_ids)
    plt.legend()
    
    # Annotate bar values
    for bar, r in zip(bars, radii):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f'{r:.4f}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )
        
    plt.tight_layout()
    plt.savefig('hasil_analisis.png', dpi=300)
    print("\n[INFO] Analysis figure saved as 'hasil_analisis.png'")
    plt.show()

# =========================
# 5) MAIN FUNCTION
# =========================

def main(csv_path: str = "data.csv"):
    # (A) File Check
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found!")
        return

    print(f"[1/5] Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # (B) Preprocessing: Aggregation and Pivoting
    # Average prices if multiple sellers sell the same commodity on the same day
    daily_price = df.groupby(["tanggal", "komoditas"])["harga_idr"].mean().reset_index()
    price_wide = daily_price.pivot(index="tanggal", columns="komoditas", values="harga_idr")
    
    # Fill missing values (forward-fill then backward-fill)
    price_wide = price_wide.ffill().bfill()
    
    # Compute log-returns
    returns_wide = price_wide.apply(compute_log_returns, axis=0)
    
    # (C) Feature Extraction & Vector Space Construction
    print("[2/5] Extracting commodity feature vectors...")
    feature_rows = []
    commodities = list(returns_wide.columns)
    
    for c in commodities:
        feats = extract_features_from_series(returns_wide[c])
        feature_rows.append(feats)
        
    feat_df = pd.DataFrame(feature_rows, index=commodities)
    
    # Standardization (Z-score) → Feature matrix X
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values)
    
    # (D) Clustering
    print("[3/5] Performing K-Means clustering...")
    k = choose_k_by_silhouette(X, k_min=2, k_max=6)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    feat_df['cluster'] = labels
    
    # (E) Stability Analysis (Eigenvalues)
    print("[4/5] Analyzing dynamical system stability...")
    results = []
    for cid in range(k):
        members = feat_df[feat_df['cluster'] == cid].index.tolist()
        
        S_matrix = build_cluster_state_series(returns_wide, members)
        A = estimate_var1_matrix(S_matrix)
        rho, eigs = spectral_radius_and_eigs(A)
        
        results.append(ClusterStabilityResult(
            cluster_id=cid,
            n_members=len(members),
            members=members,
            A=A,
            eigvals=eigs,
            spectral_radius=rho,
            stable=(rho < 1.0)
        ))
        
    # (F) Output & Visualization
    print("\n" + "=" * 40)
    print(f"ANALYSIS RESULTS (Total {len(commodities)} Commodities)")
    print("=" * 40)
    
    for res in results:
        status = "STABLE (Convergent)" if res.stable else "UNSTABLE (Divergent)"
        print(f"Cluster {res.cluster_id}: {res.n_members} Commodities")
        print(f"  > Spectral Radius (ρ) : {res.spectral_radius:.5f}")
        print(f"  > Status              : {status}")
        print(f"  > Members (sample)    : {res.members[:5]} ...")
        print("-" * 20)
        
    visualize_results(X, labels, feat_df, results)
    print("[5/5] Finished.")

if __name__ == "__main__":
    main("data.csv")