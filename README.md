# Employee Salary Prediction — ANN Regression

A deep learning regression project that predicts employee salaries using an Artificial Neural Network (ANN), with automated hyperparameter tuning via **Optuna**.

##  Pipeline Overview

### 1. Import Libraries
Core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow`, `optuna`, `sklearn`

### 2. Load Data
Reads `Salary_Data.csv` into a DataFrame and performs an initial inspection.

### 3. Handle Missing Values
- **Numeric columns** → filled with column mean
- **Categorical columns** → filled with column mode

### 4. Data Cleaning & Standardization
- Fixes inconsistent labels in `Education Level` (e.g. `"phD"` → `"PhD"`, `"Master's"` → `"Master's Degree"`)

### 5. Feature Engineering
- `is_mid_career` — binary flag for employees aged 30–45
- `edu_rank` — ordinal encoding of education level (High School → PhD)
- `modal_edu_in_title` — median education rank per Job Title (group-level feature)

### 6. Outlier Capping
IQR-based capping (not removal) applied to all continuous numeric columns; boxplots visualized before and after.

### 7. Encoding
`pd.get_dummies()` applied to remaining categorical columns with `drop_first=True`.

### 8. Scaling
`StandardScaler` fitted on input features and saved for reuse in other notebooks or deployment.

### 9. Train / Test Split
80/20 split with `random_state=42`.

### 10. Hyperparameter Tuning with Optuna
Optuna searches over:
- Number of units in layer 1 and layer 2
- Optimizer (`adam`, `sgd`, `rmsprop`, `adagrad`)
- Learning rate (log-uniform)
- Epochs and batch size

20 trials, maximizing R² score on the test set.

### 11. Build Best Model
Reconstructs the ANN using the best hyperparameters found by Optuna.

### 12. Evaluate
Custom `evaluate()` function reports R² score on both train and test sets in a formatted DataFrame.



## 📊 Model Architecture

```
Input Layer
   ↓
Dense(n_units, activation='relu')
   ↓
Dense(n_units, activation='relu')
   ↓
Dense(1)   ← regression output
```



## 📈 Metric

| Metric | Description |
|--------|-------------|
| R² Score | Proportion of variance explained by the model (higher = better) |
| MAE | Mean Absolute Error (used as loss during training) |

---

