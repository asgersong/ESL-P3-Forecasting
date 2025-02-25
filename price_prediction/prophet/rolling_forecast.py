import os
import time
import json
import numpy as np
import itertools
from matplotlib import pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("forecasts", exist_ok=True)

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
    model.add_country_holidays(country_name="ES")  # Add Spanish holidays
    for regressor in regressors:
        model.add_regressor(regressor)
    return model


def train_rolling(
    forecast_horizon=10, forecast_period=24, model_config={}, regressors=[]
):
    # Rolling forecast setup
    train_data = df_train.copy()
    test_data = df_test.copy()
    test_data = test_data.iloc[
        : min(forecast_horizon * forecast_period, len(test_data))
    ]
    all_forecasts = []
    mae_list, rmse_list, r2_list = [], [], []
    log_results = []
    model = None
    iteration = 0
    logfile_name = f"logs/forecast_log_{time.strftime('%Y%m%d_%H%M%S')}.json"

    with tqdm(
        total=len(test_data) // forecast_period
        + (len(test_data) % forecast_period > 0),
        desc="Forecasting",
        leave=False,
    ) as pbar, open(logfile_name, "w", encoding="utf-8") as f:
        while not test_data.empty:
            # Create and train model
            model = create_model(model_config, regressors)
            model.fit(train_data)

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

    return model, df_forecasts, mae_list, rmse_list, r2_list


def plot(model, forecast_log, prior_horizon=24, forecast_horizon=24):
    model.plot(forecast_log, ylabel="Price", xlabel="Time")

    plt.xlim(
        forecast_log["ds"].min() - pd.Timedelta(hours=prior_horizon),
        forecast_log["ds"].min() + pd.Timedelta(hours=forecast_horizon),
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Select ex-regressors (all columns except "ds", "y", and "price_day_ahead")
    regressors = df_train.columns.difference(["ds", "y", "price_day_ahead"])

    # model_config = {
    #     "daily_seasonality": "auto",
    #     "changepoint_prior_scale": 0.5,  # default 0.05
    # }

    # model, df_forecasts, mae_list, rmse_list, r2_list = train_rolling(
    #     forecast_horizon=1, model_config=model_config, regressors=regressors
    # )

    # # # Print overall metrics
    # print(f"Overall MAE: {np.mean(mae_list):.4f}")
    # print(f"Overall RMSE: {np.mean(rmse_list):.4f}")
    # print(f"Overall R2: {np.mean(r2_list):.4f}")

    # # save the model and forecast
    # with open("saved_models/model.json", "w", encoding="utf-8") as fout:
    #     json.dump(model_to_json(model), fout)

    # df_forecasts.to_csv("forecasts/df_forecasts.csv")

    # model_config_grid = {
    #     "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
    #     "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
    #     "holidays_prior_scale": [0.01, 0.1, 1.0, 10.0],
    # }

    # # Generate all combinations of parameters
    # all_params = [
    #     dict(zip(model_config_grid.keys(), v))
    #     for v in itertools.product(*model_config_grid.values())
    # ]
    # rmses = []
    # maes = []
    # r2s = []

    # for params in tqdm(all_params, desc="Grid search", colour="green"):
    #     model, df_forecasts, mae_list, rmse_list, r2_list = train_rolling(
    #         forecast_horizon=1, model_config=params, regressors=regressors
    #     )
    #     rmses.append(np.mean(rmse_list))
    #     maes.append(np.mean(mae_list))
    #     r2s.append(np.mean(r2_list))

    #     # save the model and forecast with unique name
    #     params_str = "_".join(f"{key}_{value}" for key, value in params.items())
    #     with open(
    #         f"saved_models/model_{params_str}.json", "w", encoding="utf-8"
    #     ) as fout:
    #         json.dump(model_to_json(model), fout)

    #     df_forecasts.to_csv(f"forecasts/df_forecasts_{params_str}.csv")

    # best_params = all_params[np.argmin(rmses)]
    # print(f"Best parameters: {best_params}")
    # print(f"Best RMSE: {np.min(rmses)}")
    # print(f"Best MAE: {maes[np.argmin(rmses)]}")
    # print(f"Best R2: {r2s[np.argmin(rmses)]}")

    # plot the forecast
    best_params_str = "_changepoint_prior_scale_0.001_seasonality_prior_scale_0.1"
    model = model_from_json(
        json.load(open(f"saved_models/model{best_params_str}.json", "r", encoding="utf-8"))
    )
    forecast_log = pd.read_csv(f"df_forecasts{best_params_str}.csv", parse_dates=["ds"])
    plot(model, forecast_log, 24 * 1, 24 * 1)
