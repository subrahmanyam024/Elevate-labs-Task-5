# Elevate-labs-Task-5

# Heart Disease Classification with Decision Trees and Random Forests

## Overview
This script (`heart_disease_tree_models.py`) performs binary classification on the `heart.csv` dataset to predict heart disease (1 = disease, 0 = no disease) using Decision Tree and Random Forest classifiers. It preprocesses the data, trains models, visualizes the Decision Tree inline in Jupyter Notebook, analyzes overfitting, evaluates Random Forest performance, interprets feature importances, and performs cross-validation.

## Dataset
- **Source**: `heart.csv` (Heart Disease dataset)
- **Target**: `target` (1 = heart disease, 0 = no disease)
- **Features**: 13 numerical/categorical features:
  - `age`: Age of the patient
  - `sex`: Sex (1 = male, 0 = female)
  - `cp`: Chest pain type
  - `trestbps`: Resting blood pressure
  - `chol`: Serum cholesterol
  - `fbs`: Fasting blood sugar
  - `restecg`: Resting electrocardiographic results
  - `thalach`: Maximum heart rate achieved
  - `exang`: Exercise-induced angina
  - `oldpeak`: ST depression induced by exercise
  - `slope`: Slope of the peak exercise ST segment
  - `ca`: Number of major vessels colored by fluoroscopy
  - `thal`: Thalassemia type

## Requirements
- Python 3.6+
- Libraries:
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn graphviz


Graphviz System Binaries (for Decision Tree Visualization):
Download from https://graphviz.org/download/ (Windows installer or ZIP).
Add Graphviz bin directory (e.g., C:\Program Files\Graphviz\bin) to system PATH.
Verify with dot -V in Command Prompt (should output dot - graphviz version X.XX.X).


Dataset: heart.csv in the same directory as the script.
Jupyter Notebook: For inline visualization and interactive plotting.

Script Description
The script performs the following steps:

Preprocessing:
Loads heart.csv using pandas.
Splits into features (X) and target (y).
Performs 80/20 train-test split (random_state=42).
Standardizes features using StandardScaler (uses transform for test data to avoid data leakage).


Decision Tree:
Trains a DecisionTreeClassifier (random_state=42).
Visualizes the tree inline in Jupyter using graphviz.Source with IPython.display.display.


Overfitting Analysis:
Computes train/test accuracy for the default Decision Tree (Train: 1.0000, Test: 0.9854).
Trains a pruned tree (max_depth=5) to reduce overfitting (Train: 0.9293, Test: 0.8439).


Random Forest:
Trains a RandomForestClassifier (100 trees, random_state=42).
Computes test accuracy (0.9854), precision (1.0000), recall (0.9709), and confusion matrix.
Visualizes the confusion matrix as a heatmap.


Feature Importances:
Extracts and plots Random Forest feature importances, identifying key predictors (e.g., cp: 0.135, ca: 0.127, thalach: 0.122, oldpeak: 0.122).


Cross-Validation:
Performs 5-fold cross-validation for both models (Decision Tree: 0.9756 ± 0.0370, Random Forest: 0.9817 ± 0.0318).



Usage

Place heart.csv in the same directory as heart_disease_tree_models.py.
Install dependencies:pip install pandas numpy scikit-learn matplotlib seaborn graphviz


Install Graphviz system binaries (for Decision Tree visualization):
Download from https://graphviz.org/download/.
Add Graphviz bin directory to PATH (e.g., C:\Program Files\Graphviz\bin).
Verify with dot -V in Command Prompt.


Run in Jupyter Notebook:
Start Jupyter:jupyter notebook


Open the script or paste into a cell and run.
Ensure inline plotting with:%matplotlib inline


The Decision Tree will display inline in the notebook.


If visualization fails, see Troubleshooting for Graphviz setup.

Outputs

Console:
Decision Tree train/test accuracies (default: 1.0000/0.9854, pruned: 0.9293/0.8439).
Random Forest test accuracy (0.9854), precision (1.0000), recall (0.9709), and confusion matrix:[[102   0]
 [  3 100]]


Feature importances (sorted by importance, e.g., cp: 0.135, ca: 0.127).
5-fold cross-validation accuracies (Decision Tree: 0.9756 ± 0.0370, Random Forest: 0.9817 ± 0.0318).


Visualizations:
Decision Tree: Displayed inline in Jupyter using Graphviz.
Random Forest Confusion Matrix: Heatmap displayed inline.
Feature Importances: Bar plot displayed inline.



Troubleshooting

Graphviz Error (ExecutableNotFound):
Cause: Graphviz system binaries (dot.exe) are not installed or not in PATH.
Fix:
Install Graphviz from https://graphviz.org/download/.
Add bin directory (e.g., C:\Program Files\Graphviz\bin) to PATH:
Right-click Start > System > Advanced system settings > Environment Variables.
Edit Path under System/User variables, add the bin path, and save.


Verify with dot -V in Command Prompt.
Restart Jupyter Notebook (Ctrl+C in terminal, then jupyter notebook) or kernel.
Check PATH in Jupyter:import os
print(os.environ['PATH'])

Ensure the Graphviz bin directory appears.
If Graphviz fails, consider adding a fallback to sklearn.tree.plot_tree (see Recommendations).




Plots Not Displaying in Jupyter:
Run %matplotlib inline in a cell before the script.
Alternatively, try:import matplotlib
matplotlib.use('TkAgg')
plt.show()




FileNotFoundError for heart.csv:
Ensure heart.csv is in the working directory (check with os.getcwd()).


Other Errors:
Verify dependencies with pip list.
Check for zero-variance features:std = X_train.std()
zero_var_cols = std[std == 0].index
if len(zero_var_cols) > 0:
    print(f"Zero-variance columns: {zero_var_cols}")
    X_train = X_train.drop(columns=zero_var_cols)
    X_test = X_test.drop(columns=zero_var_cols)





Notes

The Decision Tree is visualized inline using graphviz.Source and IPython.display.display, suitable for Jupyter Notebook.
Random Forest outperforms the Decision Tree (test accuracy: 0.9854 vs. 0.8439 for pruned tree) due to ensemble learning.
Feature importances highlight cp, ca, thalach, and oldpeak as key predictors.
The script assumes heart.csv is in the working directory and Graphviz is properly configured.

Recommendations

Hyperparameter Tuning: Use GridSearchCV to optimize max_depth, min_samples_split, etc., for Decision Tree or n_estimators, max_features for Random Forest.
Additional Metrics: Add ROC-AUC or F1-score for evaluation.
Visualization Fallback: If Graphviz errors persist, modify the script to use sklearn.tree.plot_tree:from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=['No Disease', 'Disease'], 
          filled=True, rounded=True, fontsize=10)
plt.show()


Explore Other Visualizations: Use dtreeviz for enhanced Decision Tree visualizations.


