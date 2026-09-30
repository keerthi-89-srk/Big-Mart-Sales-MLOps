from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": rmse,
        "R2 Score": r2_score(y_test, predictions),
    }


def train_and_compare(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """Train multiple regression models and compare their evaluation metrics."""
    linear_model = LinearRegression().fit(X_train, y_train)
    ridge_model = Ridge(alpha=1.0).fit(X_train, y_train)
    decision_tree_model = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    parameter_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        parameter_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    tuned_random_forest_model = grid_search.best_estimator_

    models = {
        "Linear Regression": linear_model,
        "Ridge Regression": ridge_model,
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model,
        "Tuned Random Forest": tuned_random_forest_model,
    }

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)

    comparison = pd.DataFrame.from_dict(results, orient="index")
    comparison = comparison[["MAE", "RMSE", "R2 Score"]]

    best_model_name = comparison["R2 Score"].idxmax()
    best_model = models[best_model_name]

    return comparison, best_model_name, best_model
