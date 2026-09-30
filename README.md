# Big Mart Sales Prediction

This project was refactored from a Jupyter notebook into a Python package structure for easier development, testing, and deployment.

## Project structure

- `src/bigmart_sales/` — reusable Python package containing the data preprocessing, modeling, and training logic.
- `Train.csv` and `Test.csv` — source datasets used for training and evaluation.
- `artifacts/` — generated model metrics and trained model files.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python -m bigmart_sales.train
   ```

## Model workflow

The project does the following:

- loads the Big Mart sales datasets
- cleans the data and engineers new features
- encodes categorical features
- trains multiple regression models
- compares their performance using MAE, RMSE, and R²
- saves the best model to `artifacts/final_model.pkl`

## Main commands

```bash
python -m bigmart_sales.train --train-path Train.csv --test-path Test.csv --output-dir artifacts
```
