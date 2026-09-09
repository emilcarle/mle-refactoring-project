# IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import skops.io as sio
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD DATA
def load_data(path: str = "King_County_House_prices_dataset.csv"):
    df = pd.read_csv(path)
    return df

# MISSING VALUES AND ERRONEOUS DATA
def calc_and_clean_by_bath_bed_ratio(kc_data: pd.DataFrame):
    df = kc_data.copy()
    df["bath_bed_ratio"] = df["bathrooms"] / df["bedrooms"]
    df = df[(df["bath_bed_ratio"] < 2) & (df["bath_bed_ratio"] > 0.1)]
    df = df.drop(columns="bath_bed_ratio")
    return df

def calc_sqft_basement(kc_data: pd.DataFrame):
    df = kc_data.copy()
    df["sqft_basement"] = df["sqft_living"] - df["sqft_above"]
    return df

def fillna_in_view_waterfront(kc_data: pd.DataFrame):
    df = kc_data.copy()
    columns = ['view', 'waterfront']
    for column_name in columns:
        df[column_name] = df[column_name].fillna(0)
    return df

def calc_last_known_change(kc_data: pd.DataFrame):
    df = kc_data.copy()
    last_known_change = []
    for idx, yr_re in df["yr_renovated"].items():
        if str(yr_re) == "nan" or yr_re == 0.0:
            last_known_change.append(df["yr_built"][idx])
        else:
            last_known_change.append(int(yr_re))
    df["last_known_change"] = last_known_change
    df.drop("yr_renovated", axis=1, inplace=True)
    df.drop("yr_built", axis=1, inplace=True)
    return df

def pipeline_clean(kc_data: pd.DataFrame):
    df = kc_data.copy()
    df = calc_and_clean_by_bath_bed_ratio(df)
    df = calc_sqft_basement(df)
    df = fillna_in_view_waterfront(df)
    df = calc_last_known_change(df)
    return df

# FEATURE ENGINEERING
def calc_sqft_price(kc_data: pd.DataFrame):
    df = kc_data.copy()
    df["sqft_price"] = (
        df.price / (df.sqft_living + df.sqft_lot)
    ).round(2)
    return df

def calc_center_distance(kc_data: pd.DataFrame, ref_lat: float = 47.62774, ref_long: float = -122.24194):
    df = kc_data.copy()
    df["delta_lat"] = np.absolute(ref_lat - df["lat"])
    df["delta_long"] = np.absolute(ref_long - df["long"])
    df["center_distance"] = (
        (
            (df["delta_long"] * np.cos(np.radians(ref_lat))) ** 2
            + df["delta_lat"] ** 2
        )
        ** (1 / 2)
        * 2
        * np.pi
        * 6378
        / 360
    )
    return df

def helper_dist(long, lat, ref_long, ref_lat):
    delta_long = long - ref_long
    delta_lat = lat - ref_lat
    delta_long_corr = delta_long * np.cos(np.radians(ref_lat))
    return (
        ((delta_long_corr) ** 2 + (delta_lat) ** 2) ** (1 / 2) * 2 * np.pi * 6378 / 360
    )
    
def calc_waterfront_distance(kc_data: pd.DataFrame):
    df = kc_data.copy()
    water_list = df.query("waterfront == 1")
    water_distance = []
    for idx in df.index:
        ref_list = []
        for x, y in zip(list(water_list["long"]), list(water_list["lat"])):
            ref_list.append(helper_dist(df["long"][idx], df["lat"][idx], x, y).min())
        water_distance.append(min(ref_list))
    df["water_distance"] = water_distance
    return df

def pipeline_feature_engineering(kc_data: pd.DataFrame):
    df = kc_data.copy()
    df = calc_sqft_price(df)
    df = calc_center_distance(df)
    df = calc_waterfront_distance(df)
    return df

# MODELLING
def prepare_modelling_split(kc_data: pd.DataFrame):
    df = kc_data.copy()
    X = df.drop(
        columns=["id", "price", "date", "delta_lat", "delta_long", "sqft_price"],
        errors="ignore",
    )
    y = df.price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def run_model(X: pd.DataFrame, y: pd.DataFrame, model_path: str = "model/model.bin"):
    df_x = X.copy()
    df_y = y.copy()
    untrusted_types = sio.get_untrusted_types(file=model_path)
    model = sio.load(model_path, trusted=untrusted_types)
    df_x["predicted_price"] = model.predict(df_x)
    df = pd.concat([df_x, df_y.rename("price")], axis=1)
    df["absolute_error"] = (df["predicted_price"] - df["price"]).abs()
    df["percentage_error"] = (
        df["absolute_error"] / df["price"] * 100
    ).round(2)
    r2 = r2_score(df["price"], df["predicted_price"])
    mae = mean_absolute_error(df["price"], df["predicted_price"])
    rmse = mean_squared_error(
        df["price"],
        df["predicted_price"],
    ) ** 0.5
    print(f"R²: {r2:.3f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    return df

def main() -> None:
    kc_data = load_data()
    kc_data = pipeline_clean(kc_data)
    kc_data = pipeline_feature_engineering(kc_data)
    X_train, X_test, y_train, y_test = prepare_modelling_split(kc_data)
    result = run_model(X=X_test, y=y_test)

if __name__ == "__main__":
    main()