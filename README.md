# akinesia_fingerprint_article
Contains code used for the analysis and figure creation of the article "Uncovering the Neural Fingerprint of Akinetic States in a Parkinson's Disease Rodent Model".


The code uses the toolbox <img src="https://github.com/ELGarulli/neurokin/blob/main/docs/favicon.png" height="20" alt="neurokin logo"> [![neurokin](https://img.shields.io/badge/neurokin-package-blue)](https://github.com/ELGarulli/neurokin) , which is also available on PyPi:

    pip install neurokin

The 3D printing files for the recording cap are availabe [in this repo.](https://github.com/ELGarulli/headstage_cap)
### Dataset Preparation and import
Code for the computation of kinematic and neural features is in [dataset_preparation](./dataset_preparation). The dataset used is published on Zenodo, [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15163493.svg)](https://doi.org/10.5281/zenodo.15163493).

In each script, the dataset is imported with the custom function data_load from the file [dataset_load](./dataset_preparation/dataset_load.py)

### Base analysis:
- Fig 1b: [Tyrosine Hydroxylase quantification](th_staining/th_staining.ipynb)
- Fig 1f, import/export Linear Mixed Model: [states analysis](linear_mixed_model/LMM_PSD.ipynb). The LMM computation was done in R with the formula: $$model <- lmer( beta_band_value ~ event_type + (1 | subject), data = df_long )$$
- Fig 1g, Fig 1h: [PSD and related Boxplot](linear_mixed_model/boxplot_PSD.ipynb)

### LDA analysis
Code for:
- Fig 2b, Fig 2d, Fig 2e: [LDA kinemtaic analysis](lda_analysis/LDA_quantificaiton_kin.ipynb)
- Fig 2f, Fig 2h, Fig 2i: [LDA neural analysis](lda_analysis/LDA_quantificaiton.ipynb)

### CEBRA analysis
Code for:
- Fig 3a and Fig 3e - CEBRA embeddings using kinematic or neural features: [cebra kinematic embedding](cebra_analysis/full_run/main_plot/cebra_neural.ipynb)
- Fig 3b and Fig 3f - Boxplot of most relevant features: [neural features](cebra_analysis/importance_scorer/quantification.ipynb), [kinematic features](cebra_analysis/importance_scorer/quantification_kin.ipynb). Embeddings are computed in [compute embeddings](cebra_analysis/importance_scorer/embeddings/compute_train_embedding.ipynb) and KNN classifications are computed launching the [knn_eval shell script](cebra_analysis/importance_scorer/knn_eval.sh)
- Fig 3c and Fig 3g - Boxplot of most relevant features: [neural features](cebra_analysis/full_run/heatmap/scatter_neu.ipynb), [kinematic features](cebra_analysis/full_run/heatmap/scatter_kin.ipynb).
- Fig 3d and Fig 3h - Confusion Matrixes are built and plotted here: [neural](cebra_analysis/neural_folds_cm/nerual_visualization.ipynb), [kinematics](cebra_analysis/kinematic_folds_cm/kin_visualization.ipynb). And the 11-fold cross-validation runs are computed in: [neural](cebra_analysis/neural_folds_cm/neural_run_offset.py) and [kinematics](cebra_analysis/kinematic_folds_cm/kinematic_run_offset.py) called from the respective shell scripts.
- Frames for Supplementary Video 2: [animation](cebra_analysis/animation/animation_walk_runwise.ipynb)

### Important features
Code for:
- Fig 4a: [scatterplot](lda_analysis/LDA_x_CEBRA.ipynb)
- Fig 4b: [examples and formulas](hjort_viz/hjorth_viz.ipynb)
- Fig 4c, Fig 4d: [average temporal evolution of features](modulated_features/avg_neural_time_evo.ipynb)
- Fig 4e: [group comparison](modulated_features/group_comparison.ipynb)
- Fig 4f: Embedding using [only beta](cebra_analysis/discovered_feat_comparison/nerual_visualization_beta.ipynb), embedding using only [Hjorth complexity](cebra_analysis/discovered_feat_comparison/nerual_visualization_complex.ipynb) and related .py
