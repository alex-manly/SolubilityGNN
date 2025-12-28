# Graph Neural Networks for Molecular Property Prediction

This project explores the use of graph neural networks (GNNs) to predict molecular properties (solubility) from molecular structure.

The dataset used is from AqSolDB:

Sorkun, M.C., Khetan, A. & Er, S. AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. Sci Data 6, 143 (2019). https://doi.org/10.1038/s41597-019-0151-1

## Completed Goals

- Build, from scratch, a pipeline that takes molecules and creates a featurized (via RDKit) graph representation for use in GNN architectures
- Use PyTorch and PyTorch-geometric to define a message-passing GNN with variable architecture
- Combine molecular graph pipeline and GNN model to train against AqSolDB data
- Evaluate the impact of architectural and optimization hyperparameters
- Evaluate the ability of the GNN to predict molecule solubility from atom/bond level descriptors

## Future Goals

- Benchmark GNN performance against non-message passing model architectures
- Maximize model performance by combining molecular-level features with message-passing network
- Determine the importance of atomic and bond features for predicting molecular solubility using ablation studies
- Attempt transfer learning with a foundational model to improve outlier/general predictive performance

## Repository Structure
```
├── data/
│   ├── clean-dataset.csv
│   └── curated_solubility-dataset.csv
│
├── notebooks/
│   ├── 01_DataAndGraphs.ipynb
│   └── 02_GNNAndHyperparameterTuning.ipynb
│
├── results/
│   ├── history/
│   │   ├── <model-training-histories.csv>
│   │   └── ...
│   ├── models/
│   │   ├── <model-parameter-states.pt>
│   │   └── ...
│   └── grid_search.csv
│
├── src/
│   └── SolGnn
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── GNN.py
│       ├── graphs.py
│       ├── search.py
│       ├── train.py
│       └── viz.py
│
├── pyproject.toml
└── README.md
```
- **data/**: Raw data from AqSolDB, and a cleaned dataset used for model training and evaluation
- **notebooks/**: This is where my tests are run, and results are visualized and analyzed  
- **results/**: Model training histories, parameters, and metrics from hyperparameter tests
- **src/**: Contains code that is imported by, and discussed in the notebooks 
## How to Read This Project

My results and analysis live in my notebooks. They build upon each other, and are numbered in that order.

1. **01_DataAndGraphs.ipynb**  
   Solubility data is loaded and cleaned. A reproducible data-split method is introduced. My molecule &rarr; featurized graph representation pipeline is explained and demonstrated.

2. **02_GNNAndHyperparameterTuning.ipynb**  
   My GNN architecture is inspected and discussed. Training parameters are chosen. Model hyperparameters are tested are optimized. The trained and optimized GNN model is tested against unseen data, and the results are presented/analyzed.
