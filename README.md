# Tree Tensor Networks for Hyperspectral Image Classification of Mineral Data

Research implementation for investigating Tree Tensor Networks (TTNs) for hyperspectral image classification of mineral data.

## Abstract

Hyperspectral imaging provides detailed spectral information that can be used to identify and classify minerals. However, the high dimensionality of hyperspectral data presents challenges for efficient feature representation and classification.

This project investigates Tree Tensor Networks as a framework for representing and classifying hyperspectral mineral data. The repository contains preprocessing, clustering, spectral matching, and classical machine-learning baseline pipelines, with tensor-network-based models being progressively integrated.

## Research Objectives

- Develop a preprocessing pipeline for hyperspectral mineral data.
- Extract meaningful spectral and spatial features.
- Investigate Tree Tensor Networks for hyperspectral classification.
- Compare different tensor-network architectures.
- Compare tensor-network models against classical machine-learning baselines.
- Evaluate classification accuracy and computational efficiency.

## Methodology

The overall pipeline consists of hyperspectral preprocessing, clustering, spectral matching, feature and patch generation, classification, and evaluation.

### Preprocessing

- Spectral band selection and filtering
- Data normalization
- Hyperspectral feature preparation
- Patch generation

### Clustering and Spectral Analysis

- K-Means clustering
- ISODATA clustering
- DBSCAN clustering
- Spectral Angle Mapper (SAM)

### Classification

A Support Vector Machine (SVM) is used as a classical machine-learning baseline. The tensor-network experiments investigate multiple architectures:

- Binary Tree Tensor Network
- Ternary Tree Tensor Network
- 4-ary Tree Tensor Network
- Matrix Product State / Tensor Train
- Tensor Ring

## Repository Structure

    .
    ├── configs/
    │   └── experiment.yaml
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── README.md
    ├── docs/
    │   └── methodology.md
    ├── notebooks/
    │   ├── 01_preprocessing/
    │   │   ├── kmeans_clustering.ipynb
    │   │   ├── isodata_clustering.ipynb
    │   │   └── dbscan_sam_pipeline.ipynb
    │   ├── 02_baselines/
    │   │   └── svm_baseline.ipynb
    │   └── 03_tensor_networks/
    │       └── README.md
    ├── src/
    ├── scripts/
    │   └── run_pipeline.py
    ├── results/
    │   ├── figures/
    │   └── tables/
    ├── tests/
    │   └── test_structure.py
    ├── requirements.txt
    ├── CITATION.cff
    ├── LICENSE
    └── README.md

## Experimental Comparison

| Model | Type | Status |
|---|---|---|
| K-Means | Clustering | Available |
| ISODATA | Clustering | Available |
| DBSCAN | Clustering | Available |
| SAM | Spectral Matching | Available |
| SVM | Classical ML | Available |
| Binary TTN | Tensor Network | In Progress |
| Ternary TTN | Tensor Network | In Progress |
| 4-ary TTN | Tensor Network | In Progress |
| MPS / Tensor Train | Tensor Network | In Progress |
| Tensor Ring | Tensor Network | In Progress |

## Evaluation

The classification models will be evaluated using:

- Overall Accuracy (OA)
- Per-class Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Additional analysis may include model size, training time, inference time, and computational efficiency.

## Installation

    git clone https://github.com/<username>/tree-tensor-networks-hyperspectral-mineral-classification.git
    cd tree-tensor-networks-hyperspectral-mineral-classification
    python -m venv venv

Activate the environment on Windows:

    venv\Scripts\activate

Activate the environment on Linux/macOS:

    source venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

## Running the Pipeline

The preprocessing and baseline experiments are available under `notebooks/`.

The complete pipeline can be executed using:

    python scripts/run_pipeline.py

## Data

Raw hyperspectral datasets are not included in the repository.

Place the required datasets under:

    data/raw/

Processed datasets should be stored under:

    data/processed/

Refer to `data/README.md` for the expected data organization.

## Reproducibility

Experiment parameters are maintained in:

    configs/experiment.yaml

Generated figures and quantitative results are stored under:

    results/

The repository is organized to separate datasets, preprocessing, model implementations, experiments, configurations, and results to support reproducible research.

## Research Status

- [x] Hyperspectral preprocessing
- [x] K-Means clustering
- [x] ISODATA clustering
- [x] DBSCAN clustering
- [x] Spectral Angle Mapper
- [x] SVM baseline
- [ ] Binary Tree Tensor Network
- [ ] Ternary Tree Tensor Network
- [ ] 4-ary Tree Tensor Network
- [ ] MPS / Tensor Train
- [ ] Tensor Ring
- [ ] Unified model comparison
- [ ] Final experimental analysis

## Citation

    @article{shastry_ttn_hyperspectral,
      title   = {Tree Tensor Networks for Hyperspectral Image Classification of Mineral Data},
      author  = {Arjun Shastry},
      year    = {2026}
    }

Citation details will be updated when the research paper is finalized.

## License

This project is intended for academic and research purposes. See `LICENSE` for details.
