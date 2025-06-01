import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import LabelEncoder

def parse_price(x):
    """
    Cleans up price strings like '$131.00' -> 131.0
    Returns NaN if not valid.
    """
    if pd.isnull(x):
        return np.nan
    # Remove anything not digit or dot
    clean_str = re.sub(r"[^0-9.]", "", str(x))
    if clean_str == "":
        return np.nan
    return float(clean_str)

def main():
    # 1) LOAD THE CSV
    csv_path = os.path.join("data", "airbnb_listings.csv")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # 
    # 2) DATA CLEANING / PARSING
    # 
    # Parse "price"
    df["price"] = df["price"].apply(parse_price)

    # Some Airbnb datasets store bathrooms as "bathrooms_text" (e.g., '1 bath').

    if "bathrooms_text" in df.columns and "bathrooms" not in df.columns:
        df["bathrooms"] = df["bathrooms_text"].str.extract(r"(\d+\.?\d*)").astype(float)

    # so we can use it as a numeric feature. (If not relevant, remove.)
    if "room_type" in df.columns:
        # drop rows with missing room_type
        df = df.dropna(subset=["room_type"]).copy()
        le = LabelEncoder()
        df["room_type_encoded"] = le.fit_transform(df["room_type"])
        # e.g., Private room -> 1, Entire home/apt -> 0, etc.

    # 
    # 3) OPTIONAL: HANDLE PRICE OUTLIERS
    df = df[df["price"] > 0]  # Remove zero/negative
    df = df[df["price"] < 10000]  # Remove extreme outliers
    print(f"After outlier removal, {len(df)} rows remain.")

    # 4) SELECT FEATURES FOR PRICE PREDICTION
    features_for_price = ["accommodates", "bathrooms", "bedrooms", "latitude", "longitude"]
    if "room_type_encoded" in df.columns:
        features_for_price.append("room_type_encoded")

    # Ensure "price" is there
    features_for_price.append("price")

    # Drop any rows missing these columns
    df_price = df[features_for_price].dropna().copy()

    # Separate features/target
    X = df_price.drop("price", axis=1)
    y = df_price["price"]

    # 5) TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 6) TRAIN 3 REGRESSION MODELS

    # --- (A) LINEAR REGRESSION ---
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    print(f"[LR] RMSE: {rmse_lr:.2f}, R^2: {r2_lr:.2f}")

    # --- (B) RANDOM FOREST ---
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"[RF] RMSE: {rmse_rf:.2f}, R^2: {r2_rf:.2f}")

    # --- (C) XGBOOST ---
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"[XGB] RMSE: {rmse_xgb:.2f}, R^2: {r2_xgb:.2f}")

    # 
    # 7) SAVE EACH REGRESSION MODEL
    # 
    joblib.dump(lr_model, "model_lr.pkl")
    joblib.dump(rf_model, "model_rf.pkl")
    joblib.dump(xgb_model, "model_xgb.pkl")
    print("Saved LR (model_lr.pkl), RF (model_rf.pkl), XGB (model_xgb.pkl)")

    # 



























































    
    # 8) K-MEANS CLUSTERING
    # 
    # We'll cluster on some location/capacity columns: latitude, longitude, accommodates

    cluster_cols = ["latitude", "longitude", "accommodates"]
    df_cluster = df[cluster_cols].dropna().copy()

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(df_cluster)

    # Save the KMeans model
    joblib.dump(kmeans, "kmeans.pkl")
    print("Saved K-Means (kmeans.pkl)")

if __name__ == "__main__":
    main()
