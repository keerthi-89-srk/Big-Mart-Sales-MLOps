import pandas as pd
import pytest

from src.preprocess import preprocess_data


def _training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Item_MRP": [10.0, None, 30.0],
            "Item_Fat_Content": ["LF", "Regular", None],
            "Item_Outlet_Sales": [100.0, 200.0, 300.0],
        }
    )


def _test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Item_MRP": [None, 40.0],
            "Item_Fat_Content": ["reg", "Low Fat"],
        }
    )


def test_preprocess_data_imputes_encodes_and_preserves_input():
    train_df = _training_data()
    test_df = _test_data()
    train_original = train_df.copy(deep=True)
    test_original = test_df.copy(deep=True)

    X_train, X_test, y_train = preprocess_data(train_df, test_df)

    assert X_train.columns.equals(X_test.columns)
    assert X_train.notna().all().all()
    assert X_test.notna().all().all()
    assert all(dtype == "float64" for dtype in X_train.dtypes)
    assert y_train.tolist() == [100.0, 200.0, 300.0]
    assert train_df.equals(train_original)
    assert test_df.equals(test_original)
    assert "Item_Fat_Content_Low Fat" in X_train.columns
    assert "Item_Fat_Content_Regular" in X_train.columns


def test_preprocess_data_handles_pandas_category_features():
    train_df = pd.DataFrame(
        {
            "Item_Type": pd.Series(["Dairy", None], dtype="category"),
            "Item_Outlet_Sales": [1.0, 2.0],
        }
    )
    test_df = pd.DataFrame({"Item_Type": pd.Series(["Baking"], dtype="category")})

    X_train, X_test, _ = preprocess_data(train_df, test_df)

    assert X_train.columns.equals(X_test.columns)
    assert "Item_Type_Baking" in X_train.columns
    assert X_train.notna().all().all()
    assert X_test.notna().all().all()


def test_preprocess_data_rejects_mismatched_feature_columns():
    train_df = _training_data()
    test_df = _test_data().rename(columns={"Item_MRP": "Different"})

    with pytest.raises(ValueError, match="feature columns do not match"):
        preprocess_data(train_df, test_df)


def test_preprocess_data_rejects_missing_target():
    train_df = _training_data().drop(columns=["Item_Outlet_Sales"])

    with pytest.raises(ValueError, match="missing target column"):
        preprocess_data(train_df, _test_data())


@pytest.mark.parametrize(
    "target",
    [
        [100.0, None, 300.0],
        [100.0, float("inf"), 300.0],
        ["100", "not-a-number", "300"],
    ],
)
def test_preprocess_data_rejects_invalid_target_values(target):
    train_df = _training_data()
    train_df["Item_Outlet_Sales"] = target

    with pytest.raises(ValueError, match="Target column"):
        preprocess_data(train_df, _test_data())


def test_preprocess_data_rejects_all_missing_numeric_feature():
    train_df = pd.DataFrame(
        {
            "Item_MRP": pd.Series([float("nan")], dtype="float64"),
            "Item_Outlet_Sales": [100.0],
        }
    )
    test_df = pd.DataFrame(
        {"Item_MRP": pd.Series([float("nan")], dtype="float64")}
    )

    with pytest.raises(ValueError, match="Item_MRP contains only missing values"):
        preprocess_data(train_df, test_df)


def test_preprocess_data_rejects_non_finite_numeric_feature():
    train_df = pd.DataFrame(
        {"Item_MRP": [float("inf")], "Item_Outlet_Sales": [100.0]}
    )
    test_df = pd.DataFrame({"Item_MRP": [10.0]})

    with pytest.raises(ValueError, match="Item_MRP contains non-finite values"):
        preprocess_data(train_df, test_df)


@pytest.mark.parametrize(
    ("train_df", "test_df", "message"),
    [
        (
            pd.DataFrame(columns=["Item_MRP", "Item_Outlet_Sales"]),
            pd.DataFrame(columns=["Item_MRP"]),
            "Training dataset is empty",
        ),
        (
            _training_data(),
            pd.DataFrame(columns=["Item_MRP", "Item_Fat_Content"]),
            "Test dataset is empty",
        ),
        (
            _training_data(),
            _test_data().assign(Item_Outlet_Sales=0.0),
            "must not contain target column",
        ),
    ],
)
def test_preprocess_data_rejects_invalid_dataset_shapes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        preprocess_data(train_df, test_df)
