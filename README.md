# Iris Dataset Classification with Apache Spark MLlib
### STQD6324 Data Management — Assignment 1 | P162501 | Semester 2, 2025/2026

---

## Project Overview

This project implements a complete **Apache Spark MLlib classification pipeline** on the classic Iris dataset.  
Three algorithms — **Decision Tree**, **Random Forest**, and **Logistic Regression** — are trained,  
systematically tuned using **5-fold cross-validation with grid search**, and evaluated on a held-out test set.  
A full comparative analysis is conducted to justify the best-performing model.

---

## Dataset

| Property | Detail |
|----------|--------|
| Name | Iris Dataset |
| Source | `sklearn.datasets.load_iris()` |
| Samples | 150 (50 per class) |
| Features | 4 numeric: sepal length, sepal width, petal length, petal width (all in cm) |
| Classes | 3: Setosa, Versicolor, Virginica |
| Missing values | None |

---

## Methodology

### Pipeline Steps

| Step | Component | Detail |
|------|-----------|--------|
| 1 | Environment Setup | PySpark session, auto JAVA_HOME detection |
| 2 | Data Loading | scikit-learn → pandas → Spark DataFrame |
| 3 | EDA | Schema, statistics, class distribution, missing values, scatter + pairplots |
| 4 | Preprocessing | `StringIndexer` (label encoding) + `VectorAssembler` (feature vector) |
| 5 | Train-Test Split | 80% train / 20% test, seed=42, both sets cached |
| 6 | Evaluation Setup | Accuracy, Weighted Precision, Weighted Recall, Weighted F1 |
| 7 | Model Tuning | 5-fold `CrossValidator` + `ParamGridBuilder`, `parallelism=4` |
| 8 | Test Evaluation | Unbiased metrics on held-out test set |
| 9 | Predictions | Species name mapping + probability scores |
| 10 | Comparative Analysis | Bar chart, confusion matrices, feature importances, best model justification |

### Hyperparameter Grids

| Model | Parameters Tuned | Grid Size |
|-------|-----------------|-----------|
| Decision Tree | `maxDepth` {3,5,7}, `maxBins` {16,32}, `minInstancesPerNode` {1,2,5} | 18 |
| Random Forest | `numTrees` {10,50,100}, `maxDepth` {3,5,7}, `maxBins` {16,32} | 18 |
| Logistic Regression | `regParam` {0,0.01,0.1,0.5}, `elasticNetParam` {0,0.5}, `maxIter` {100,200} | 16 |

---

## Summary of Results

All three models achieve strong performance on the Iris dataset:

- **Petal length** and **petal width** are the most discriminative features (confirmed by RF feature importance)
- **Setosa** is perfectly classified by all models — it is linearly separable in petal space
- Misclassifications occur only between **Versicolor** and **Virginica**, reflecting their natural feature overlap
- The **best model** is selected dynamically by weighted F1-score (see Step 10.5 in the notebook)

### Model Comparison

| Model | Strengths | Limitations |
|-------|-----------|-------------|
| Decision Tree | Interpretable IF-ELSE rules; no scaling required | Prone to overfitting; high variance |
| Random Forest | Robust ensemble; built-in feature importance | Less interpretable; higher compute cost |
| Logistic Regression | Fast; calibrated probabilities; interpretable coefficients | Assumes linear boundaries |

---

## How to Reproduce

### Prerequisites

```bash
pip install pyspark scikit-learn pandas numpy matplotlib seaborn
```

Java 11 or 17 must be installed. The notebook auto-detects `JAVA_HOME` for common Linux/Colab paths.  
For Windows, set manually: `os.environ['JAVA_HOME'] = 'C:/Program Files/Java/jdk-11'`

### Run

```bash
jupyter notebook P162501_STQD6324_Assignment1.ipynb
```

Select **Kernel → Restart & Run All** to execute all cells in order.

---

## Output Files Generated

| File | Description |
|------|-------------|
| `eda_scatter.png` | Sepal and petal scatter plots by species |
| `eda_pairplot.png` | All 6 feature-pair combinations |
| `model_comparison.png` | Bar chart: Accuracy, Precision, Recall, F1 across all models |
| `confusion_matrices.png` | Side-by-side confusion matrix heatmaps for all 3 models |
| `feature_importance.png` | Random Forest Gini feature importance ranking |

---

## Repository Structure

```
iris-spark-mllib/
├── P162501_STQD6324_Assignment1.ipynb   ← Main notebook
├── README.md                             ← This file
├── eda_scatter.png
├── eda_pairplot.png
├── model_comparison.png
├── confusion_matrices.png
└── feature_importance.png
```

---

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.8 |
| PySpark | ≥ 3.5 |
| Java | 11 or 17 |
| scikit-learn | any recent |
| pandas | any recent |
| numpy | any recent |
| matplotlib | any recent |
| seaborn | any recent |
