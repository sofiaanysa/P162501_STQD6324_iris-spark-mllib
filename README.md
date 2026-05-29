# Iris Dataset Classification with Apache Spark MLlib

| | |
|---|---|
| **Name** | SOFIA ANNISA BINTI JAIRI |
| **Student ID** | P162501 |
| **Course** | STQD6324 Data Management — Assignment 1 (25%) |
| **Semester** | 2, 2025/2026 |
| **Submission Deadline** | 2026-05-31 |

---

## Project Overview

This project implements a complete **multi-class classification pipeline** on the classic **Iris dataset** using **Apache Spark MLlib**. The goal is to classify iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on four morphological measurements.

Three classification algorithms are implemented and compared:
- **Logistic Regression** (multinomial)
- **Decision Tree**
- **Random Forest**

Each model is systematically tuned via **5-fold cross-validation** and **grid search** to find optimal hyperparameters. Final models are evaluated using **accuracy**, **weighted precision**, **weighted recall**, and **F1-score**, followed by a detailed comparative analysis.

---

## Repository Structure

```
P162501_STQD6324_iris-spark-mllib/
├── P162501_STQD6324_Iris_SparkMLlib.ipynb   # Main Jupyter Notebook
├── Iris.csv                                  # Dataset (also loaded via URL)
└── README.md                                 # This file
```

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | Iris Dataset |
| **Source** | [GitHub (project repo)](https://raw.githubusercontent.com/sofiaanysa/P162501_STQD6324_iris-spark-mllib/main/Iris.csv) |
| **Samples** | 150 (50 per class — perfectly balanced) |
| **Features** | SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm |
| **Target** | Species: Iris-setosa (0.0), Iris-versicolor (1.0), Iris-virginica (2.0) |
| **Format** | CSV with header |

The dataset is well-studied and balanced, making accuracy a reliable primary metric. All four features are continuous numeric values requiring no imputation or scaling for tree-based models.

---

## Methodology

### 1. Environment Setup
Install PySpark, pandas, matplotlib, seaborn, and requests. Initialise a Spark session.

### 2. Data Loading
Load the Iris CSV from a public URL into a Spark DataFrame using `inferSchema=True`. Inspect schema, first 5 rows, basic statistics, and class distribution.

### 3. Data Preprocessing
- **Label Encoding:** Convert the `Species` string column to numeric indices (0.0, 1.0, 2.0) using `StringIndexer`.
- **Feature Assembly:** Combine the four feature columns into a single `features` vector column using `VectorAssembler` — the required input format for Spark ML estimators.

### 4. Train-Test Split
An **80/20 stratified-equivalent split** using `randomSplit([0.8, 0.2], seed=123)` yields 121 training and 29 test samples. The fixed seed ensures full reproducibility.

### 5. Model Definition
Three classifiers are initialised with baseline settings:
- **Logistic Regression** — `family="multinomial"`, `maxIter=100`
- **Decision Tree** — default settings
- **Random Forest** — `numTrees=50`, `seed=123`

### 6. Hyperparameter Tuning
**5-fold cross-validation** combined with **grid search** (`CrossValidator` + `ParamGridBuilder`) is applied to each model. Parallelism is set to 4 for efficiency.

| Model | Parameter | Search Space |
|---|---|---|
| Logistic Regression | `regParam` | 0.01, 0.1, 1.0 |
| | `elasticNetParam` | 0.0 (L2), 0.5 (ElasticNet), 1.0 (L1) |
| | `maxIter` | 10, 20, 50 |
| Decision Tree | `maxDepth` | 3, 5, 10 |
| | `maxBins` | 8, 16, 32 |
| | `minInstancesPerNode` | 1, 2, 5 |
| Random Forest | `numTrees` | 10, 20, 50 |
| | `maxDepth` | 3, 5, 10 |
| | `maxBins` | 8, 16, 32 |

**Optimal hyperparameters found:**
- **LR:** `regParam=0.01`, `elasticNetParam=0.0` (pure L2/Ridge), `maxIter=10`
- **DT:** `maxDepth=5`, `maxBins=32`, `minInstancesPerNode=1`
- **RF:** `numTrees=20`, `maxDepth=3`, `maxBins=16`

### 7. Evaluation
A reusable `evaluate_model()` function computes all four metrics using `MulticlassClassificationEvaluator`. Confusion matrices are visualised per model using seaborn heatmaps.

---

## Results Summary

### Performance Metrics (on Test Set, n=29)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **0.9655** | **0.9690** | **0.9655** | **0.9649** |
| Decision Tree | 0.9310 | 0.9310 | 0.9310 | 0.9310 |
| **Random Forest** | **0.9655** | **0.9690** | **0.9655** | **0.9649** |

### Confusion Matrix Summary

| Model | Setosa | Versicolor | Virginica | Errors |
|---|---|---|---|---|
| Logistic Regression | 14/14 ✅ | 5/6 | 9/9 ✅ | 1 (Versicolor → Virginica) |
| Decision Tree | 14/14 ✅ | 5/6 | 8/9 | 2 (cross-species) |
| Random Forest | 14/14 ✅ | 5/6 | 9/9 ✅ | 1 (Versicolor → Virginica) |

All misclassifications occur between **Versicolor** and **Virginica**, reflecting the natural overlap in their petal and sepal measurements — a well-known challenge in this dataset.

---

## Key Findings

- **Logistic Regression** and **Random Forest** tie at the highest accuracy (0.9655), each making only 1 error on the 29-sample test set.
- **Decision Tree** achieves 0.9310, with 2 errors — slightly less stable on this small dataset.
- All models perfectly classify **Iris-setosa**, which is linearly separable from the other two species.
- The Versicolor/Virginica boundary is the main challenge for all algorithms.

---

## Model Comparison

### Logistic Regression ✅ Best Model
**Strengths:**
- Matches the highest accuracy (0.9655) with the lowest complexity.
- Fast to train and produces well-calibrated probability estimates per class.
- Fully interpretable: coefficients directly reflect feature influence.
- Pure L2 regularisation (`elasticNetParam=0.0`) prevents overfitting without eliminating features.

**Limitations:**
- Assumes linear decision boundaries — may underperform on highly non-linear data.
- Sensitive to feature scaling (though not an issue here as all features are on similar scales).

### Decision Tree
**Strengths:**
- Explicitly captures non-linear boundaries via split rules.
- Fully interpretable as a visual tree structure.
- No feature scaling required.

**Limitations:**
- Slightly lower accuracy (0.9310) on this test set.
- Prone to overfitting on small datasets; individual trees can be unstable.

### Random Forest
**Strengths:**
- Matches Logistic Regression in accuracy (0.9655); the ensemble reduces the variance of individual trees.
- Robust to noise and outliers; less prone to overfitting than a single Decision Tree.
- Feature importance scores are available as a bonus diagnostic.

**Limitations:**
- Higher computational cost — 20 trees vs. 1 model.
- Less interpretable than a single tree or Logistic Regression.
- Overkill for a linearly separable dataset like Iris.

### Justification of Best Model
**Logistic Regression** is selected as the best model for this task. It achieves the highest accuracy tied with Random Forest, but does so with far less computational cost and greater interpretability. Given that the Iris dataset is largely linearly separable (especially Setosa), and the remaining Versicolor/Virginica boundary can be well-approximated by a linear boundary in the feature space, Logistic Regression is the most appropriate and efficient choice.

---

## How to Reproduce

### Prerequisites

- Python 3.8+
- Java 8 or 11 (required by PySpark)
- pip

### Option A: Run in Google Colab (Recommended)

1. Open the notebook directly in [Google Colab](https://colab.research.google.com/).
2. Upload `P162501_STQD6324_Iris_SparkMLlib.ipynb`, or open from GitHub.
3. Run all cells in order (`Runtime → Run all`).
4. No additional setup is needed — the first cell installs all dependencies.

### Option B: Run Locally

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/sofiaanysa/P162501_STQD6324_iris-spark-mllib.git
cd P162501_STQD6324_iris-spark-mllib
```

**Step 2 — Install dependencies:**
```bash
pip install pyspark pandas matplotlib seaborn requests jupyter
```

**Step 3 — Launch Jupyter Notebook:**
```bash
jupyter notebook P162501_STQD6324_Iris_SparkMLlib.ipynb
```

**Step 4 — Run all cells in order.**

> ⚠️ Ensure Java is installed and `JAVA_HOME` is set correctly before running PySpark.

### Reproducibility Notes
- All random operations use `seed=123` for consistent results.
- The dataset is fetched automatically from a public GitHub URL — no manual download needed.
- The notebook runs end-to-end without errors; expected outputs are embedded in each cell.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pyspark` | Distributed ML framework (Spark MLlib) |
| `pandas` | Tabular data display and comparison table |
| `matplotlib` | Plotting (bar charts) |
| `seaborn` | Confusion matrix heatmaps |
| `requests` | Fetching the dataset from a URL |

---

## Notebook Structure

| Section | Description |
|---|---|
| 1. Environment Setup | Install packages, import libraries, start Spark session |
| 2. Data Loading | Load CSV, inspect schema, statistics, class counts |
| 3. Data Preprocessing | Label encoding (`StringIndexer`), feature assembly (`VectorAssembler`) |
| 4. Train-Test Split | 80/20 split with `seed=123` |
| 5. Model Definition | Initialise LR, DT, RF classifiers |
| 6. Evaluator Setup | Define `MulticlassClassificationEvaluator` |
| 7. Model Tuning | 5-fold CV + grid search for each model |
| 8. Model Evaluation | Compute accuracy, precision, recall, F1 on test set |
| 9. Confusion Matrices | Heatmap visualisation per model |
| 10. Sample Predictions | Show top-5 test predictions with probabilities |
| 11. Comparative Analysis | Performance table, bar chart, model strengths/limitations |
| 12. Conclusion | Summary of findings and best model justification |
| 13. Stop Spark Session | Clean up resources |
