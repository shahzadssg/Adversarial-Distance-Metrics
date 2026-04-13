# Adversarial Distance Metrics: A Threat to Fairness in Clustering-Based Decision Systems

This repository contains the implementation and experimental code for the paper *Adversarial Distance Metrics: A Threat to Fairness in Clustering-Based Decision Systems*.

## Overview

We demonstrate that clustering algorithms (K-means, DBSCAN, agglomerative) can be manipulated to produce **arbitrary, predetermined outcomes** by replacing the standard Euclidean distance with a carefully constructed **ε-semimetric**---a distance function that satisfies all metric properties except for an arbitrarily small violation of the triangle inequality.

The attack preserves data integrity and algorithmic transparency. An auditor inspecting the code sees a standard clustering algorithm operating on unaltered data with a legitimate-looking distance function. Yet outcomes can be fully controlled by the adversary.

### Key Results

| Experiment | Algorithm | Attack Accuracy | Construction Time |
|---|---|---|---|
| Synthetic (m=130) | K-means | 100% | ~3 hours |
| Synthetic (m=50) | K-means | 100% | 23 seconds |
| Synthetic (m=130) | DBSCAN | 100% | ~3 hours |
| Adult Income (m=10) | K-means | 100% | 0.026 seconds |

On real census data (UCI Adult Income), the attack increases Demographic Parity Difference from 0.200 to **1.000** (theoretical maximum), forcing all White individuals into the favorable cluster and all Black individuals into the unfavorable cluster.

## Repository Structure

```
├── compute_metric.m              # Core: constructs the ε-semimetric (matrix A and scale s)
├── epsilon_semimetric.m          # Core: evaluates the ε-semimetric for a given point pair
├── kmeans_customDist.m           # K-means implementation accepting a custom distance function
├── dbscan_customDist.m           # DBSCAN implementation accepting a custom distance function
├── benchmark_metric.m            # Benchmarks metric construction time for varying m
├── benchmark_clustering.m        # Benchmarks full attack (construction + clustering)
├── benchmark_linind.m            # Tests linear independence of projected data points
├── benchmark_linind_b0.m         # Linear independence benchmark variant (offset=0)
├── benchmark_linind_fix1.m       # Linear independence benchmark with fix 1
├── benchmark_linind_fix2.m       # Linear independence benchmark with fix 2
├── experiment_adult_income.m     # Real-world fairness experiment (UCI Adult Income)
└── saved_variables/              # Auto-created directory for computed metric matrices
```

## Requirements

- **MATLAB R2023a or later** (tested on R2025a)
- Statistics and Machine Learning Toolbox (for baseline `kmeans`)
- Symbolic Math Toolbox (only for variable precision experiments)

## Quick Start

### 1. Synthetic Data Attack

Construct an adversarial metric for m=20 points in 2D with 3 clusters:

```matlab
% Construct the metric (saves A and s to saved_variables/)
benchmark_metric(20, 2, "double");

% Run the full attack with K-means
benchmark_clustering(20, "double", "kmeans");

% Run the full attack with DBSCAN
benchmark_clustering(20, "double", "dbscan");
```

### 2. Real-World Fairness Attack (Adult Income)

```matlab
% Ensure saved_variables/ exists and all .m files are in the same directory
experiment_adult_income
```

This runs the full pipeline:
1. Loads 10 individuals from UCI Adult Income (5 White, 5 Black)
2. Runs baseline Euclidean K-means and measures Demographic Parity Difference
3. Constructs an ε-semimetric targeting race-based discrimination
4. Runs adversarial K-means and DBSCAN, reporting attack accuracy and DPD

Expected output: 100% K-means attack accuracy, DPD = 1.000.

### 3. Benchmark Construction Time

```matlab
% Time the metric construction for increasing sample sizes
for m = [10 20 30 40 50]
    fprintf('m = %d: ', m);
    benchmark_metric(m, 2, "double");
end
```

## How It Works

### ε-Semimetric Construction (`compute_metric.m`)

Given a dataset Y = {y₁, ..., yₘ} ⊂ ℝˡ and desired pairwise distances δᵢⱼ:

1. **Embed** points into ℝʰ where h = C(m,2) by appending deterministic pseudorandom noise
2. **Compute** difference vectors Δᵢⱼ = zᵢⱼ − zⱼᵢ for all pairs
3. **QR decomposition** of the matrix of difference vectors
4. **Construct** a positive semidefinite matrix A such that √(Δᵢⱼᵀ A Δᵢⱼ) = δᵢⱼ for all pairs

The resulting function d̃(x, y) = ‖(x + f(x,y)) − (y + f(y,x))‖_A is an ε-semimetric: it satisfies identity, positivity, symmetry, and an approximate triangle inequality with violation ≤ ε.

### Evaluation (`epsilon_semimetric.m`)

At runtime, for any pair (x, y):

1. Generate deterministic noise vectors from coordinate-derived seeds
2. Embed into ℝʰ: zₓ = [x, s·noise₁], z_y = [y, s·noise₂]
3. Return √((zₓ − z_y)ᵀ A (zₓ − z_y))

### Attack Pipeline

1. Adversary chooses target cluster assignments (e.g., race-based)
2. Sets δᵢⱼ = small for same-cluster pairs, large for cross-cluster pairs
3. Calls `compute_metric` to build the ε-semimetric (one-time cost)
4. Configures the clustering algorithm to use the crafted distance function
5. Clustering produces the predetermined outcome

## Computational Complexity

| Phase | Theoretical | Empirical | Bottleneck |
|---|---|---|---|
| Metric construction | O(m⁸) | O(m⁶) | QR decomposition of h null spaces |
| K-means execution | O(km) | Negligible | One-iteration convergence |
| DBSCAN execution | O(m²) | Dominated by construction | Pairwise distance queries |

Empirical scaling: t ≈ 7 × 10⁻⁹ · m⁵·⁷² seconds (double precision, Intel i7-6820HQ).

## Known Limitations

- **Numerical conditioning.** For real-world data with repeated feature values, the matrix A can become ill-conditioned (cond(A) > 10¹⁵). The current workaround is to add micro-jitter (order 10⁻⁸) to break seed collisions in the pseudorandom noise generator.
- **Scalability.** Double precision: m ≤ 160. Variable precision: m ≤ 38 (with ~1000× overhead). Large-scale attacks are computationally infeasible.
- **Distance constraint.** Desired distances must satisfy δᵢⱼ ≤ ‖yᵢ − yⱼ‖₂. Cross-cluster distances get clamped to 0.99 × Euclidean distance when this is violated, which can affect DBSCAN threshold selection.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{adversarial_distance_metrics2026,
  title     = {Adversarial Distance Metrics: A Threat to Fairness in Clustering-Based Decision Systems},
  author    = {Anonymous},
  booktitle = {Proceedings of SECRYPT 2026},
  year      = {2026}
}
```

This work builds on the mathematical framework of ε-semimetrics introduced in:

```bibtex
@article{rass2024,
	author = {Rass, Stefan and K{\"o}nig, Sandra and Ahmad, Shahzad and Goman, Maksim},
	title = {Metricizing the Euclidean Space Toward Desired Distance Relations in Point Clouds},
	journal = {IEEE Transactions on Information Forensics and Security},
	volume = {19},
	pages = {7304--7319},
	year = {2024},
	doi = {10.1109/TIFS.2024.3420246}
}
```

## License

The custom K-means implementation (`kmeans_customDist.m`) is licensed under the GNU Affero General Public License v3.0 (see file header). All other code is provided for research purposes.
