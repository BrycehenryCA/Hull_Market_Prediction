# Hull Tactical – Market Prediction

### 1. Project Overview

This project implements a baseline machine learning approach for the Hull Tactical Market Prediction challenge. It uses historical market data to predict market forward excess returns and generate portfolio allocation decisions.

The solution uses a LightGBM regression model with time-series cross-validation, rolling statistical features, and an ensemble of trained models to produce test-set predictions.

### 2. Technologies and Libraries

The project uses the following technologies:

- Python 3
- Pandas – Data manipulation and analysis
- NumPy – Numerical computations
- LightGBM – Gradient boosting regression
- Scikit-learn – Data preprocessing, time-series validation, and evaluation
- Matplotlib – Data visualization
- Jupyter Notebook – Model development and experimentation

### 3. Dataset

The project uses two datasets:

| File | Description |
|---|---|
| `train.csv` | Historical training data containing market features and target variables |
| `test.csv` | Test data used to generate market allocation predictions |

Dataset dimensions:

- Training data: 8,990 rows and 98 columns.
- Testing data: 10 rows and 99 columns.

**Target variable:** `market_forward_excess_returns`

**Output variable:** `allocation`

### 4. Project Workflow

The notebook follows these main stages:

1. **Data Loading:** Import the training and testing datasets using Pandas.
2. **Data Preprocessing:** Remove constant columns and handle missing values using forward filling and median imputation.
3. **Feature Engineering:** Generate rolling means and standard deviations using 5-day, 10-day, and 20-day windows.
4. **Feature Preparation:** Separate the target variable from the input features and prepare training and testing matrices.
5. **Feature Scaling:** Apply StandardScaler to standardize numerical features.
6. **Model Training:** Train LightGBM regression models using five-fold TimeSeriesSplit cross-validation.
7. **Model Evaluation:** Evaluate prediction performance using RMSE and a custom pseudo-Sharpe metric.
8. **Feature Importance:** Identify the most influential features using LightGBM gain-based importance.
9. **Prediction:** Average predictions from the trained models to generate final test predictions.
10. **Portfolio Allocation:** Convert predictions into allocation values and clip them to the valid range of 0 to 2.
11. **Submission:** Export the final predictions to `submission.csv`.

### 5. Model Configuration

The baseline model uses LightGBM with the following parameters:

| Parameter | Value |
|---|---|
| Objective | Regression |
| Boosting type | GBDT |
| Number of leaves | 64 |
| Learning rate | 0.01 |
| Feature fraction | 0.8 |
| Bagging fraction | 0.8 |
| Bagging frequency | 5 |
| Maximum boosting rounds | 8,000 |
| Early stopping rounds | 200 |
| Cross-validation folds | 5 |
| Random seed | 42 |

### 6. Model Performance

The notebook produces the following evaluation results:

| Evaluation Metric | Result |
|---|---:|
| Final validation fold RMSE | 0.011388 |
| Local Pseudo-Sharpe Score | 0.021832 |

**Note:** The reported RMSE is from the fifth validation fold, not the average RMSE across all five folds. The pseudo-Sharpe score is a custom local evaluation metric.

### 7. Output Files

The notebook generates the following output:

- `submission.csv` – Contains predicted portfolio allocations for the test dataset.
- Feature importance visualization – Displays the top 25 influential features.
- Allocation visualization – Shows predicted portfolio allocations for the test observations.

The submission file contains two columns:

| Column | Description |
|---|---|
| `date_id` | Identifier for each test observation |
| `allocation` | Predicted portfolio allocation between 0 and 2 |

### 8. Installation

Install the required dependencies using:

```bash
pip install lightgbm pandas numpy scikit-learn matplotlib
```

### 9. Running the Project

1. Place `train.csv` and `test.csv` in the project directory.
2. Open the Jupyter Notebook containing the baseline implementation.
3. Install the required dependencies.
4. Execute the notebook cells sequentially.
5. Review the evaluation metrics and visualizations.
6. Locate the generated `submission.csv` file in the project directory.

