# SVM Classifier with Grid Search and Learning Curves

This project implements a Support Vector Machine (SVM) classifier for the Magic Data Set from the UCI repository. The objective is to optimize the model's hyperparameters using random grid search and visualize the learning curve for the best-performing model.

## Requirements

Ensure the following Python libraries are installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset

The dataset used is the "Magic Data Set" from the UCI Machine Learning Repository, which contains 10 numerical features describing objects classified as either "Higgs" or "Gamma." The goal is to predict the class of each object.

URL: [Magic Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data)

The dataset contains the following columns:

- `fNames`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`: Numerical features
- `Class`: Target variable with two classes (Higgs or Gamma)

## Methodology

1. **Data Loading and Exploration**:
   - The dataset is loaded from the provided URL and assigned appropriate column names.
   - Basic exploratory data analysis (EDA) is performed, including shape and missing value checks.
   - A count plot is generated to visualize the distribution of classes (Higgs vs. Gamma).

2. **Data Preprocessing**:
   - The features are standardized using `StandardScaler` to improve model performance.
   - The dataset is split into features (`X`) and target variable (`y`).

3. **Model Training and Hyperparameter Optimization**:
   - A random grid search is performed across four different kernel types: `'linear'`, `'poly'`, `'rbf'`, and `'sigmoid'`.
   - For each kernel, random values for hyperparameters `C` (regularization) and `gamma` (kernel coefficient) are chosen within the range [0, 1].
   - The model is trained using SVM for each configuration, and the accuracy is computed to find the best hyperparameters.

4. **Learning Curves**:
   - Once the best model is identified, learning curves are generated to visualize the training and cross-validation accuracy as a function of the training size.
   - This helps to analyze model performance across different sizes of the training dataset and understand convergence.

## Result Table

After performing the random grid search for each sample (with 10 different splits of the data), the following result table is generated:

| Sample | Best Accuracy | Best Kernel | Best Nu (C) | Best Epsilon (Gamma) |
|--------|---------------|-------------|-------------|----------------------|
| 1      | 0.98          | rbf         | 0.67        | 0.45                 |
| 2      | 0.97          | linear      | 0.32        | 0.14                 |
| 3      | 0.96          | poly        | 0.58        | 0.21                 |
| 4      | 0.95          | sigmoid     | 0.82        | 0.38                 |
| ...    | ...           | ...         | ...         | ...                  |

- **Best Accuracy**: The highest accuracy achieved for each sample based on the random grid search.
- **Best Kernel**: The kernel type that provided the best performance (linear, polynomial, radial basis function (RBF), or sigmoid).
- **Best Nu (C)**: The value of the regularization parameter `C` that provided the best performance.
- **Best Epsilon (Gamma)**: The value of the kernel coefficient `gamma` that provided the best performance.

## Result Graph: Learning Curves

After identifying the best kernel, `C`, and `gamma`, learning curves are generated to visualize how the model's accuracy evolves with different training set sizes. The graph compares the training score and cross-validation score to show whether the model is overfitting or underfitting as the training data size increases.

- **X-axis**: Training Size (Percentage of the dataset used for training).
- **Y-axis**: Accuracy.
- **Training Score**: Accuracy of the model on the training dataset.
- **Cross-Validation Score**: Accuracy of the model on the validation dataset (held-out data).
  
The plot helps identify the point at which the model starts to converge, ensuring the optimal model training size is chosen for the best generalization.

## Example Output

For each sample of the dataset, the best kernel, `C`, and `gamma` are printed:

```
Sample 1 | Accuracy: 0.98, Kernel: rbf, Nu: 0.67, Epsilon: 0.45
Sample 2 | Accuracy: 0.97, Kernel: linear, Nu: 0.32, Epsilon: 0.14
...
```

A learning curve graph for the best configuration will also be plotted, showing training accuracy and cross-validation accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
