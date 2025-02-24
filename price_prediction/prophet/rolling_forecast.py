import os
import json
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

os.makedirs("saved_models", exist_ok=True)

# Load data
df_train = pd.read_csv("../../dataset/processed/train/train.csv")
df_test = pd.read_csv("../../dataset/processed/test/test.csv")

# Rename and format columns
df_train = df_train.rename(columns={"time": "ds", "price_actual": "y"})
df_test = df_test.rename(columns={"time": "ds", "price_actual": "y"})
df_train["ds"] = pd.to_datetime(df_train["ds"], format="%Y-%m-%d %H:%M:%S")
df_test["ds"] = pd.to_datetime(df_test["ds"], format="%Y-%m-%d %H:%M:%S")


# Define function to create model
def create_model(model_config, regressors):
    model = Prophet(**model_config)
    model.add_country_holidays(country_name="ES") # Add Spanish holidays
    for regressor in regressors:
        model.add_regressor(regressor)
    return model


# Select regressors
regressors = df_train.columns.difference(["ds", "y", "price_day_ahead"])
model_config = {
    "daily_seasonality": "auto", 
    "changepoint_prior_scale": 0.5, # default 0.05
}


def train_rolling(forecast_horizon=10, forecast_period=24):
    # Rolling forecast setup
    train_data = df_train.copy()
    test_data = df_test.copy()
    test_data = test_data.iloc[:min(forecast_horizon*forecast_period, len(test_data))]
    all_forecasts = []
    mae_list, rmse_list, r2_list = [], [], []
    log_results = []
    iteration = 0

    with tqdm(total=len(test_data) // forecast_period + (len(test_data) % forecast_period > 0)) as pbar, open(
        "forecast_log.json", "w", encoding="utf-8"
    ) as f:
        while not test_data.empty:
            # Create and train model
            model = create_model(model_config, regressors)
            model.fit(train_data)

            # Save the latest model
            model_filename = f"saved_models/model_iteration_{iteration}.json"
            with open(model_filename, "w", encoding="utf-8") as fout:
                json.dump(model_to_json(model), fout)

            # Forecast next 24 hours (or remaining test data if less than 24)
            horizon = min(forecast_period, len(test_data))
            future = test_data.iloc[:horizon].copy()
            forecast = model.predict(future)

            # Store results
            y_true = future["y"].values
            y_pred = forecast["yhat"].values
            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
            r2 = r2_score(y_true, y_pred)
            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)

            log_entry = {
                "start": str(future["ds"].min()),
                "end": str(future["ds"].max()),
                "values": y_true.tolist(),
                "predictions": y_pred.tolist(),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
            log_results.append(log_entry)
            f.seek(0)
            json.dump(log_results, f, indent=4)
            f.truncate()
            f.write("\n")

            print(
                f"Forecasting {future['ds'].min()} to {future['ds'].max()} \
                    -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}"
            )

            all_forecasts.append(forecast)

            # Move forecasted data into training set
            train_data = pd.concat([train_data, future], ignore_index=True)

            # Remove used test data
            test_data = test_data.iloc[horizon:]

            iteration += 1
            pbar.update(1)

    df_forecasts = pd.concat(all_forecasts, ignore_index=True)

    # log df_forecasts
    df_forecasts.to_csv("df_forecasts.csv")

    # Print overall metrics
    print(f"Overall MAE: {np.mean(mae_list):.4f}")
    print(f"Overall RMSE: {np.mean(rmse_list):.4f}")
    print(f"Overall R2: {np.mean(r2_list):.4f}")


def plot(model_iteration=0, prior_horizon=24, forecast_horizon=24):
    # Load model
    model_filename = f"saved_models/model_iteration_{model_iteration}.json"
    with open(model_filename, "r", encoding="utf-8") as fin:
        model = model_from_json(json.load(fin))

    # Load forecast
    forecast_filename = "df_forecasts.csv"
    forecast_log = pd.read_csv(forecast_filename, parse_dates=["ds"])

    model.plot(forecast_log, ylabel="Price", xlabel="Time")

    plt.xlim(
        forecast_log["ds"].min() - pd.Timedelta(hours=prior_horizon),
        forecast_log["ds"].min() + pd.Timedelta(hours=forecast_horizon),
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # train_rolling(forecast_horizon=10)
    plot(9, 24*1, 24*10)

    # # plot the training data
    # df_train.plot(x="ds", y="y")
    # df_test.plot(x="ds", y="y")
    # plt.show()
