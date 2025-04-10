# akinesia_fingerprint_article
Contains code used for the analysis and figure creation of the article "Uncovering the Neural Fingerprint of Akinetic States in a Parkinson's Disease Rodent Model"

### Dataset Preparation and import
Code for the computation of kinematic and neural features is in [dataset_preparation](./dataset_preparation). The dataset used is published on Zenodo, [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15163493.svg)](https://doi.org/10.5281/zenodo.15163493).

In each script, the dataset is imported with the custom function data_load from the file [dataset_load](./dataset_preparation/dataset_load.py)

### CEBRA analysis
Code for:
- Fig3a and Fig3e - CEBRA embeddings using kinematic or neural features: [cebra kinematic embedding](cebra_analysis/full_run/main_plot/cebra_neural.ipynb)
- Fig3b and Fig3f - Boxplot of most relevant features: [neural features](cebra_analysis/importance_scorer/quantification.ipynb), [kinematic features](cebra_analysis/importance_scorer/quantification_kin.ipynb). Embeddings are computed in [compute embeddings](cebra_analysis/importance_scorer/embeddings/compute_train_embedding.ipynb) and KNN classifications are computed launching the [knn_eval shell script](cebra_analysis/importance_scorer/knn_eval.sh)
- Fig3c and Fig3g - Boxplot of most relevant features: [neural features](cebra_analysis/full_run/heatmap/scatter_neu.ipynb), [kinematic features](cebra_analysis/full_run/heatmap/scatter_kin.ipynb).
- Fig3d and Fig3h - Confusion Matrixes are built and plotted here: [neural](cebra_analysis/neural_folds_cm/nerual_visualization.ipynb), [kinematics](cebra_analysis/kinematic_folds_cm/kin_visualization.ipynb). And the 11-fold cross-validation runs are computed in: [neural](cebra_analysis/neural_folds_cm/neural_run_offset.py) and [kinematics](cebra_analysis/kinematic_folds_cm/kinematic_run_offset.py) called from the respective shell scripts.
- Fig4f: Embedding using [only beta](cebra_analysis/discovered_feat_comparison/nerual_visualization_beta.ipynb), embedding using only [Hjorth complexity](cebra_analysis/discovered_feat_comparison/nerual_visualization_complex.ipynb) and related .py
- Frames for Supplementary Video 2: [animation](cebra_analysis/animation/animation_walk_runwise.ipynb)
