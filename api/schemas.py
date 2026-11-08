from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BigMartInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    Item_Identifier: str = Field(min_length=1, max_length=32)
    Item_Weight: float | None = Field(ge=0, allow_inf_nan=False)
    Item_Fat_Content: Literal["Low Fat", "Regular", "LF", "low fat", "reg"]
    Item_Visibility: float = Field(ge=0, allow_inf_nan=False)
    Item_Type: Literal[
        "Baking Goods",
        "Breads",
        "Breakfast",
        "Canned",
        "Dairy",
        "Frozen Foods",
        "Fruits and Vegetables",
        "Hard Drinks",
        "Health and Hygiene",
        "Household",
        "Meat",
        "Others",
        "Seafood",
        "Snack Foods",
        "Soft Drinks",
        "Starchy Foods",
    ]
    Item_MRP: float = Field(ge=0, allow_inf_nan=False)
    Outlet_Identifier: Literal[
        "OUT010",
        "OUT013",
        "OUT017",
        "OUT018",
        "OUT019",
        "OUT027",
        "OUT035",
        "OUT045",
        "OUT046",
        "OUT049",
    ]
    Outlet_Establishment_Year: int = Field(ge=1900, le=2100)
    Outlet_Size: Literal["High", "Medium", "Small"] | None
    Outlet_Location_Type: Literal["Tier 1", "Tier 2", "Tier 3"]
    Outlet_Type: Literal[
        "Grocery Store",
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3",
    ]


class SalesPrediction(BaseModel):
    predicted_sales: float = Field(allow_inf_nan=False)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[BigMartInput] = Field(min_length=1, max_length=20)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[SalesPrediction]


class ValidationErrorDetail(BaseModel):
    field: str
    message: str


class ValidationErrorResponse(BaseModel):
    status: Literal["error"]
    message: str
    details: list[ValidationErrorDetail]
