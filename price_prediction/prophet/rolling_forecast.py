import os
import time
import json
import numpy as np
import itertools
from matplotlib import pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sympy import li
from tqdm import tqdm

os.makedirs("saved_models", exist_ok=True)
os.makedirs("forecasts", exist_ok=True)

# Load data
df_train = pd.read_csv("../../dataset/processed/train/train.csv")
df_test = pd.read_csv("../../dataset/processed/test/test.csv")

# Rename and format columns
df_train = df_train.rename(columns={"time": "ds", "price_actual": "y"})
df_test = df_test.rename(columns={"time": "ds", "price_actual": "y"})
df_train["ds"] = pd.to_datetime(df_train["ds"], format="%Y-%m-%d %H:%M:%S")
df_test["ds"] = pd.to_datetime(df_test["ds"], format="%Y-%m-%d %H:%M:%S")

# Define a time boundary for train/test split to ensure no leakage
# This is only needed if you're not already splitting correctly
train_end = df_train["ds"].max()
test_start = df_test["ds"].min()

print(f"Training data ends: {train_end}")
print(f"Test data starts: {test_start}")
print(f"Gap between train and test: {test_start - train_end}")


# remove unnecessary columns in test data
df_test_y = df_test["y"]
df_test = df_test.drop(
    columns=[
        "y",
        "price_day_ahead",
        "fossil_fuels",
        "windpower",
        "solarpower",
        "other_green_energy",
        "total_load_actual",
    ]
)


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
    mae_list, rmse_list = [], []
    model = None

    # Store original test data for creating proper next training data
    original_test_data = test_data.copy()

    num_iterations = len(test_data) // forecast_period + (
        len(test_data) % forecast_period > 0
    )
    for _ in tqdm(
        range(num_iterations), desc="Forecasting", leave=False, colour="green"
    ):
        # Create and train model
        model = create_model(model_config, regressors)
        model.fit(train_data)

        # Forecast next period
        horizon = min(forecast_period, len(test_data))
        future = test_data.iloc[:horizon].copy()
        # future = model.make_future_dataframe(periods=horizon, freq="h", include_history=False)

        # # Add regressors
        # for regressor in regressors:
        #     future[regressor] = original_test_data[regressor].iloc[:horizon].values

        print(f"History: {model.history['ds'].min()} to {model.history['ds'].max()}")
        print(f"Forecasting: {future['ds'].min()} to {future['ds'].max()} \n")

        # ensure that model.history does not contain future data
        assert future["ds"].min() > model.history["ds"].max(), "Data leakage detected"

        # make predictions
        forecast = model.predict(future)

        # Store results
        y_true, y_pred = df_test_y.loc[future.index].values, forecast["yhat"].values
        mae, rmse = (
            mean_absolute_error(y_true, y_pred),
            mean_squared_error(y_true, y_pred) ** 0.5,
        )
        mae_list.append(mae)
        rmse_list.append(rmse)

        print(
            f"Forecast from {future['ds'].min()} to {future['ds'].max()} \
                -> MAE: {mae:.4f}, RMSE: {rmse:.4f}"
        )

        all_forecasts.append(forecast)

        # Move forecasted data into training set
        train_data = pd.concat(
            [train_data, original_test_data.iloc[:horizon]], ignore_index=True
        )

        # Remove used test data
        test_data = test_data.iloc[horizon:]
        original_test_data = original_test_data.iloc[horizon:]

    df_forecasts = pd.concat(all_forecasts, ignore_index=True)

    return model, df_forecasts, mae_list, rmse_list


def plot(model: Prophet, forecast_log, prior_horizon=24, forecast_horizon=24):
    # plot history
    history = model.history.copy()
    plt.plot(
        history["ds"],
        history["y"],
        label="Actual",
        color="blue"
    )

    forecast_log["ds"] = pd.date_range(
        start=forecast_log["ds"].min(), periods=len(forecast_log), freq="h"
    )

    # plot forecast
    plt.plot(forecast_log["ds"], forecast_log["yhat"].shift(-24), 'r-', label="yhat (shifted)")
    plt.plot(forecast_log["ds"], forecast_log["yhat"], 'g-' , label="yhat (original)")

    # add uncertainty interval to the plot
    # plt.fill_between(
    #     forecast_log["ds"],
    #     forecast_log["yhat_lower"].shift(-24),
    #     forecast_log["yhat_upper"].shift(-24),
    #     color="gray",
    #     alpha=0.2,
    #     label="Uncertainty",
    # )

    # plot test data from df_test_y
    plt.plot(
        forecast_log["ds"],
        df_test_y.loc[forecast_log.index],
        label="Actual test data",
        color="blue",
    )

    plt.xlim(
        forecast_log["ds"].min() - pd.Timedelta(hours=prior_horizon),
        forecast_log["ds"].min() + pd.Timedelta(hours=forecast_horizon),
    )

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Select ex-regressors
    regressors = [
        "day_of_week",
        "hour_of_day",
        "price_actual_lag_24h",
        "price_actual_lag_1w",
        "price_actual_lag_2w",
        "price_actual_lag_3w",
        "price_actual_lag_4w",
        "fossil_fuels_lag_24h",
        "windpower_lag_24h",
        "solarpower_lag_24h",
        "other_green_energy_lag_24h",
        "total_load_actual_lag_24h",
    ]
    # regressors = []

    model_config = {
        "changepoint_prior_scale": 0.001,  # default 0.05
        "seasonality_prior_scale": 0.01,  # default 10.0
        "holidays_prior_scale": 0.01,  # default 10.0
    }

    # model, df_forecasts, mae_list, rmse_list = train_rolling(
    #     forecast_horizon=20,
    #     forecast_period=24,
    #     model_config=model_config,
    #     regressors=regressors
    # )

    # # # Print overall metrics
    # print(f"Overall MAE: {np.mean(mae_list):.4f}")
    # print(f"Overall RMSE: {np.mean(rmse_list):.4f}")

    # # save the model and forecast
    # with open("saved_models/model_new.json", "w", encoding="utf-8") as fout:
    #     json.dump(model_to_json(model), fout)

    # df_forecasts.to_csv("forecasts/df_forecasts_new.csv")

    # # plot the forecast
    # best_params_str = "_changepoint_prior_scale_0.001_seasonality_prior_scale_0.01_holidays_prior_scale_0.01"
    # suffix = "_no_ylags"
    # suffix = "_all_with_lags"
    suffix = "_new"
    # suffix = "_no_ex"
    model = model_from_json(
        json.load(open(f"saved_models/model{suffix}.json", "r", encoding="utf-8"))
    )
    forecast_log = pd.read_csv(
        f"forecasts/df_forecasts{suffix}.csv", parse_dates=["ds"]
    )
    plot(model, forecast_log, 24 * 7, 24 * 20)

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

    # # find best model in forecasts folder
    # forecast_files = os.listdir("forecasts")
    # best_rmse = float("inf")
    # best_r2 = float("-inf")
    # best_mae = float("inf")
    # best_params = None
    # y_true = df_test["y"].iloc[:24*30].values
    # for file in forecast_files:
    #     if file.startswith("df_forecasts_"):
    #         df_forecasts = pd.read_csv(f"forecasts/{file}")
    #         rmse = rmse_list = mean_squared_error(y_true, df_forecasts["yhat"]) ** 0.5
    #         r2 = r2_score(y_true, df_forecasts["yhat"])
    #         mae = mean_absolute_error(y_true, df_forecasts["yhat"])
    #         if rmse < best_rmse:
    #             best_rmse = rmse
    #             best_params = file.split("_")[2:]
    #         if r2 > best_r2:
    #             best_r2 = r2
    #             # best_params = file.split("_")[2:]
    #         if mae < best_mae:
    #             best_mae = mae
    #             # best_params = file.split("_")[2:]

    # print(f"Best RMSE: {best_rmse}")
    # print(f"Best R2: {best_r2}")
    # print(f"Best MAE: {best_mae}")
    # print(f"Best params: {best_params}")
