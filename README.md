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

## Labs 7–13 setup

Install the project and API/test dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-api-test.txt
```

Place the Big Mart `Train.csv` and `Test.csv` files in `data/raw/`, then run
`python src/preprocess.py`. This creates the processed feature and target files
under `data/processed/`; Lab 12 uses `X_train_processed.csv` with its matching
`y_train_processed.csv` for historical performance metrics. This is training
data, not an independent holdout evaluation.

The default configurations use `sqlite:///mlflow.db` and model
`BigMart_Sales_Model` version 3. A fresh clone must be supplied with that MLflow
registry database **and its corresponding artifact store**, including the
registered model artifacts. Once available, prepare the API bundle with:

```bash
python src/mlops_pipeline/10_prepare_api_bundle.py --config config/lab10_api.json
```

The generated `deployment/bigmart_sales_model_v3/` bundle is local and is
required by the API, monitoring, and automation workflows; it is not included
in Git. Lab 13's optional `--deploy` operation additionally requires the
Lab 11 `deploy_lab11.ps1` script, which is not included in this upload.
