# Graph Signal Reconstruction Based on Power Spectral Density Estimation

[![Python Version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of the computational framework proposed in the Master's diploma thesis:
**"Reconstruction_of_graph_signals_based_on_the_power_spectral_density_estimation.pdf"** by Michał Iwicki (Warsaw University of Technology, Faculty of Mathematics and Information Science).

## Overview

Graph signal reconstruction aims to recover signals defined on irregular graph domains from incomplete and noisy observations. While traditional approaches rely on deterministic smoothness assumptions (e.g., Laplacian regularization), this framework explicitly exploits second-order statistical information in the form of the **Graph Power Spectral Density (PSD)** under the assumption of Graph Wide-Sense Stationarity (GWSS). 

A major limitation of standard PSD-based methods is the assumption that all observed signals originate from a single stationary distribution. This repository introduces a novel extension: **a spectral Expectation-Maximization (EM) framework** designed to identify and reconstruct mixtures of graph signals originating from multiple latent PSD distributions. By clustering signals in the graph Fourier domain and estimating cluster-specific PSDs, the framework significantly improves reconstruction quality for heterogeneous data.

## Key Features

*   **Graph Processing & Synthesis:** Tools to generate standard graph topologies (KNN, Grid, Erdős–Rényi, Barabási–Albert, Watts–Strogatz) and precompute their Laplacian eigendecompositions.
*   **Stationary Signal Generation:** Synthesis of fully and partially observed graph signals from predefined continuous spectral profiles (e.g., low-pass, band-pass, high-pass).
*   **PSD Estimation from Partial Observations:** Robust estimation of graph PSD from randomly sampled vertex observations, employing sampled-covariance correction.
*   **Graph Signal Reconstruction:** Implementation of classical Laplacian smoothness-based reconstruction and PSD-adaptive spectral regularization.
*   **Spectral Clustering (Diagonal GMM):** A highly efficient Gaussian Mixture Model that operates on graph Fourier features. By leveraging the fact that stationary graph signals have diagonal covariance matrices in the spectral domain, this method avoids full covariance estimation, reducing parameter complexity from $O(N^2)$ to $O(N)$.
*   **Mixed Signal Workflows:** High-level pipelines (Iterative, EM-PSD) to jointly cluster partially observed signals and refine their reconstructions using learned, cluster-specific PSD splines.

##  Repository Structure

The codebase is modularized to cleanly separate data generation, core reconstruction math, clustering logic, high-level workflows, and experiments.

*   `src/methods/generation.py`: Graph topologies and synthetic data creation.
*   `src/methods/reconstruction.py`: PSD estimation and signal recovery logic.
*   `src/methods/clustering.py`: Spectral clustering (`GMM_Diag`) and evaluation metrics.
*   `src/methods/models.py`: Complete workflows combining clustering and reconstruction (`MixedSignalReconstruction`).
*   `src/experiments/`: A suite of scripts to reproduce thesis results.

## Getting Started

### Installation
Make sure you have Python 3.14+ installed. Assuming you have already downloaded the repository, simply install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Interactive Exploration
If you want to familiarize yourself with the problem of graph signal reconstruction based on PSD estimation, the best starting point is the **`preliminary_experiments.ipynb`** notebook. It provides an interactive, step-by-step introduction to graph generation, signal synthesis, and basic reconstruction, allowing you to build intuition before diving into the complex mixture models.

### Replicating Thesis Results
The repository includes a comprehensive suite of scripts used to generate the quantitative results presented in the thesis. 

You can easily run all experiments sequentially using the main execution script:
```bash
python -m src.experiments.run_all_experiments
```

This script automatically executes the following studies:
*   **Graph Size and Visibility:** Evaluates reconstruction across different numbers of nodes and varying observation probabilities.
*   **Model Improvements:** Compares the baseline clustered reconstruction against advanced iterative and simultaneous EM-PSD workflows.
*   **Topologies:** Tests reconstruction on K-Nearest-Neighbors, Grid, Erdős–Rényi, Barabási–Albert, and Watts–Strogatz graphs.
*   **Spline Transfer:** Analyzes the transferability of PSD profiles learned on a partial training graph to a full test graph (for both single and mixed signals).
*   **Multiple PSD Mixtures:** Evaluates the framework's robustness across random mixtures of different PSD profiles.

**Experiment Outputs:**
All aggregated results—including metrics like Mean Absolute Error (MAE), Adjusted Rand Index (ARI), and Clustering Accuracy—are automatically saved as `.csv` files to the `results/reconstruction` and `results/clustering` directories. Experiment parameters and configurations are logged in `results/configs` in JSON format.

## Methodology Highlights

The most significant contribution of this repository is the reformulation of the EM algorithm in the graph spectral domain. Standard Gaussian likelihood evaluation requires computing full covariance determinants and inverses, which is computationally prohibitive and numerically unstable for high-dimensional graphs.

By projecting initial smooth reconstructions into the Graph Fourier Transform (GFT) basis:
$$ \hat{f} = U^T f $$
We leverage the property that GWSS signals possess diagonal covariance matrices $\Gamma$ in this domain. The proposed E-step evaluates likelihoods using only the diagonal spectral variances, and the M-step updates simplify to computing cluster-specific PSD coefficients. This drops parameter estimation from $O(N^2)$ to $O(N)$ per mixture component, enabling highly scalable separation of signal sources.

## Citation
If you utilize this framework or code, please reference the original master's thesis:
> Iwicki, M. (2026). *Reconstruction of Graph Signals Based on Power Spectral Density Estimation*. Master's diploma thesis, Warsaw University of Technology.
