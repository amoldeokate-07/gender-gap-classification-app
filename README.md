# Gender Classification App

## Problem statement

Compare five classification algorithms on the Gender Gap in Spanish Wikipedia dataset and provide an interactive app for evaluating saved models on a labeled test CSV.

## Dataset description

The dataset is **Gender Gap in Spanish WP**, published in the [UCI Machine Learning Repository](https://www.ics.uci.edu/dataset/852/gender%2Bgap%2Bin%2Bspanish%2Bwp), DOI [10.24432/C5XP52](https://doi.org/10.24432/C5XP52). It contains 4,746 editor records. The target is `gender` (0 = unknown, 1 = male, 2 = female). The alternative label fields `C_api` and `C_man` are excluded from the predictors to prevent label leakage. The dataset has 18 remaining candidate input columns; `firstDay` and `lastDay` are converted into year, month, day, and hour features.

The data has no missing values or duplicate rows. The classes are imbalanced, so the evaluation reports macro-averaged precision, recall, F1, and one-vs-rest macro AUC in addition to accuracy and MCC. The split is stratified, with a fixed random seed for reproducibility.

## GitHub Repository Link


## Streamlit app link


