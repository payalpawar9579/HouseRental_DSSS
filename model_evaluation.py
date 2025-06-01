import pandas as pd
import numpy as np
import re
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def regression_accuracy(y_true, y_pred, threshold=0.1):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid zero-division when y_true = 0
    epsilon = 1e-9
    within_thresh = np.abs(y_pred - y_true) <= threshold * (np.abs(y_true) + epsilon)
    return np.mean(within_thresh)

def main():
    # 1) Load CSV
    # Make sure you have `airbnb_listings.csv` inside a folder named `data`.
    csv_path = os.path.join("data", "airbnb_listings.csv")
    df = pd.read_csv(csv_path, low_memory=False)

    # 2) Parse price
    def parse_price(x):
        # remove '$' or commas
        return float(re.sub(r"[^0-9.]", "", str(x))) if pd.notnull(x) else np.nan

    df["price"] = df["price"].apply(parse_price)

    # If "bathrooms_text" exists (like "1 bath"), but no "bathrooms" column,
    # parse out numeric part:
    if "bathrooms_text" in df.columns and "bathrooms" not in df.columns:
        df["bathrooms"] = df["bathrooms_text"].str.extract(r"(\d+\.?\d*)").astype(float)

    # 3) Choose columns for price prediction
    needed_cols = ["accommodates", "bathrooms", "bedrooms", "latitude", "longitude", "price"]
    df_model = df[needed_cols].dropna()

    X = df_model.drop("price", axis=1)
    y = df_model["price"]

    # 4) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) Train 3 Models for Price

    # (A) Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_r2 = r2_score(y_test, y_pred_lr)
    lr_acc = regression_accuracy(y_test, y_pred_lr, threshold=0.1)

    print(f"[LR] Accuracy: {lr_acc:.2f}, MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}, R^2: {lr_r2:.2f}")

    # (B) Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_acc = regression_accuracy(y_test, y_pred_rf, threshold=0.1)

    print(f"[RF] Accuracy: {rf_acc:.2f}, MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, R^2: {rf_r2:.2f}")

    # (C) XGBoost
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    xgb_r2 = r2_score(y_test, y_pred_xgb)
    xgb_acc = regression_accuracy(y_test, y_pred_xgb, threshold=0.1)

    print(f"[XGB] Accuracy: {xgb_acc:.2f}, MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}, R^2: {xgb_r2:.2f}")

    # 6) Save each model
    joblib.dump(lr_model, "model_lr.pkl")
    joblib.dump(rf_model, "model_rf.pkl")
    joblib.dump(xgb_model, "model_xgb.pkl")
    print("Saved model_lr.pkl, model_rf.pkl, model_xgb.pkl")

    # 7) K-Means Clustering
    # We'll pick columns like latitude, longitude, accommodates
    cluster_cols = ["latitude", "longitude", "accommodates"]
    df_cluster = df[cluster_cols].dropna()

    # Choose 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(df_cluster)

    joblib.dump(kmeans, "kmeans.pkl")
    print("Saved kmeans.pkl")

if __name__ == "__main__":
    main()