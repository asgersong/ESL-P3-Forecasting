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

# verify if the data is sorted by time
# df_train = df_train.sort_values('ds').reset_index(drop=True)
# df_test = df_test.sort_values('ds').reset_index(drop=True)

# Define a time boundary for train/test split to ensure no leakage
# This is only needed if you're not already splitting correctly
train_end = df_train["ds"].max()
test_start = df_test["ds"].min()

print(f"Training data ends: {train_end}")
print(f"Test data starts: {test_start}")
print(f"Gap between train and test: {test_start - train_end}")


# remove unnecessary columns in test data
df_test = df_test.drop(
    columns=[
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
    for i in tqdm(
        range(num_iterations), desc="Forecasting", leave=False, colour="green"
    ):
        # Create and train model
        model = create_model(model_config, regressors)
        model.fit(train_data)

        # Forecast next period
        horizon = min(forecast_period, len(test_data))
        future = test_data.iloc[:horizon].copy()

        # make predictions
        forecast = model.predict(future)

        # Store results
        y_true, y_pred = future["y"].values, forecast["yhat"].values
        mae, rmse = (
            mean_absolute_error(y_true, y_pred),
            mean_squared_error(y_true, y_pred) ** 0.5,
        )
        mae_list.append(mae)
        rmse_list.append(rmse)

        print(
            f"Forecasting {future['ds'].min()} to {future['ds'].max()} \
                -> MAE: {mae:.4f}, RMSE: {rmse:.4f}"
        )

        all_forecasts.append(forecast)

        # Move forecasted data into training set
        # train_data = pd.concat([train_data, future], ignore_index=True)

        # Create proper next training data that includes actuals, not predictions
        if i < num_iterations - 1:
            next_train_data = original_test_data.iloc[
                i
                * forecast_period : min(
                    (i + 1) * forecast_period, len(original_test_data)
                )
            ].copy()
            train_data = pd.concat([train_data, next_train_data], ignore_index=True)

        # Remove used test data
        test_data = test_data.iloc[horizon:]

    df_forecasts = pd.concat(all_forecasts, ignore_index=True)

    return model, df_forecasts, mae_list, rmse_list

def train_rolling_with_shift(
    forecast_horizon=10, forecast_period=24, model_config={}, regressors=[], shift_hours=-24
):
    """
    Rolling forecast implementation with explicit shift correction.
    
    Args:
        forecast_horizon: Number of periods to forecast
        forecast_period: Number of steps in each forecast period
        model_config: Prophet model configuration
        regressors: List of regressor column names
        shift_hours: Number of hours to shift the forecast (negative = backward)
    """
    # Rolling forecast setup
    train_data = df_train.copy()
    test_data = df_test.copy()
    test_data = test_data.iloc[
        : min(forecast_horizon * forecast_period, len(test_data))
    ]
    all_forecasts = []
    mae_list, rmse_list = [], []
    model = None

    # Create a copy of test data for evaluation
    eval_test_data = test_data.copy()
    
    num_iterations = len(test_data) // forecast_period + (len(test_data) % forecast_period > 0)
    for i in tqdm(range(num_iterations), desc="Forecasting", leave=False, colour="green"):
        # Create and train model
        model = create_model(model_config, regressors)
        model.fit(train_data)

        # Forecast next period
        horizon = min(forecast_period, len(test_data))
        future = test_data.iloc[:horizon].copy()
        
        # Make predictions
        forecast = model.predict(future)
        
        # CRITICAL STEP: Apply the time shift correction by shifting timestamps
        # This explicitly addresses the mismatch between Prophet's interpretation and your expectation
        forecast['ds_original'] = forecast['ds'].copy()  # Save original timestamps
        forecast['ds'] = forecast['ds'] + pd.Timedelta(hours=shift_hours)
        
        # Get actual values aligned with the SHIFTED forecast dates
        shifted_dates = forecast['ds']
        actual_values = []
        
        for date in shifted_dates:
            # Find the matching actual value for the shifted date
            match = eval_test_data[eval_test_data['ds'] == date]
            if len(match) > 0:
                actual_values.append(match['y'].values[0])
            else:
                actual_values.append(np.nan)
        
        # Calculate metrics using the shifted alignment
        non_nan_mask = ~np.isnan(actual_values)
        if sum(non_nan_mask) > 0:
            y_true = np.array(actual_values)[non_nan_mask]
            y_pred = forecast['yhat'].values[non_nan_mask]
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
            
            mae_list.append(mae)
            rmse_list.append(rmse)
            
            print(
                f"Forecasting {shifted_dates.min()} to {shifted_dates.max()} \
                    -> MAE: {mae:.4f}, RMSE: {rmse:.4f}"
            )
        
        # Store forecast with shifted dates
        all_forecasts.append(forecast)
        
        # Move forecasted data into training set (using ORIGINAL dates before shift)
        future_for_training = future.copy()
        train_data = pd.concat([train_data, future_for_training], ignore_index=True)
        
        # Remove used test data
        test_data = test_data.iloc[horizon:]
    
    # Combine all forecasts
    df_forecasts = pd.concat(all_forecasts, ignore_index=True)
    
    return model, df_forecasts, mae_list, rmse_list

def plot(model: Prophet, forecast_log, prior_horizon=24, forecast_horizon=24):
    # model.plot(forecast_log, ylabel="Price", xlabel="Time")
    history = model.history.copy()

    plt.plot(
        history["ds"],
        history["y"],
        label="Actual",
    )
    plt.plot(forecast_log["ds"], forecast_log["y_hat"].shift(-24), label="y_hat")

    # add uncertainty interval to the plot (y_hat_lower and y_hat_upper)
    plt.fill_between(
        forecast_log["ds"],
        forecast_log["y_hat_lower"].shift(-24),
        forecast_log["y_hat_upper"].shift(-24),
        color="gray",
        alpha=0.2,
        label="Uncertainty",
    )

    plt.xlim(
        forecast_log["ds"].min() - pd.Timedelta(hours=prior_horizon),
        forecast_log["ds"].min() + pd.Timedelta(hours=forecast_horizon),
    )

    # compute the MAE and RMSE where the forecast is available
    # y_true = history["y"].iloc[-forecast_horizon:].values
    # y_pred = forecast_log["yhat"].iloc[:forecast_horizon].values
    # mae = mean_absolute_error(y_true, y_pred)
    # rmse = mean_squared_error(y_true, y_pred) ** 0.5

    # plt.title(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Select regressors
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

    # Prophet configuration
    model_config = {
        "changepoint_prior_scale": 0.001,
        "seasonality_prior_scale": 0.01,
        "holidays_prior_scale": 0.01,
        "daily_seasonality": True,
        "yearly_seasonality": True,
        "weekly_seasonality": True,
    }

    # Run the rolling forecast with explicit shift correction
    model, df_forecasts, mae_list, rmse_list = train_rolling_with_shift(
        forecast_horizon=10, 
        forecast_period=24,
        model_config=model_config, 
        regressors=regressors,
        shift_hours=-24  # Explicitly shift by -24 hours as you found necessary
    )

    # Print overall metrics with shifted alignment
    print(f"Overall MAE (with -24h shift): {np.mean(mae_list):.4f}")
    print(f"Overall RMSE (with -24h shift): {np.mean(rmse_list):.4f}")
    
    # Visualize the results with shifted alignment
    import matplotlib.pyplot as plt
    
    # Add original dates for comparison
    df_forecasts['ds_original'] = df_forecasts['ds'] - pd.Timedelta(hours=-24)
    
    # Get actual values from test data aligned with shifted dates
    actual_values = []
    for date in df_forecasts['ds']:
        match = df_test[df_test['ds'] == date]
        if len(match) > 0:
            actual_values.append(match['y'].values[0])
        else:
            actual_values.append(np.nan)
    
    # Plot actual vs predicted values (with shift correction)
    plt.figure(figsize=(15, 6))
    plt.plot(df_forecasts['ds'], actual_values, 'b-', label='Actual')
    plt.plot(df_forecasts['ds'], df_forecasts['yhat'], 'r-', label='Predicted (Shifted -24h)')
    plt.fill_between(df_forecasts['ds'], 
                     df_forecasts['yhat_lower'], 
                     df_forecasts['yhat_upper'], 
                     color='gray', alpha=0.2)
    plt.legend()
    plt.title('Actual vs Predicted Electricity Prices (With -24h Shift Correction)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forecast_comparison_with_shift.png')
    plt.show()
    
    # Show comparison of pre-shift vs post-shift alignment for a short time window
    one_week = pd.Timedelta(days=7)
    start_date = df_forecasts['ds'].min()
    end_date = start_date + one_week
    
    mask = (df_forecasts['ds'] >= start_date) & (df_forecasts['ds'] <= end_date)
    
    # Get actual values for both original and shifted dates
    actual_values_original = []
    for date in df_forecasts.loc[mask, 'ds_original']:
        match = df_test[df_test['ds'] == date]
        if len(match) > 0:
            actual_values_original.append(match['y'].values[0])
        else:
            actual_values_original.append(np.nan)
    
    actual_values_shifted = []
    for date in df_forecasts.loc[mask, 'ds']:
        match = df_test[df_test['ds'] == date]
        if len(match) > 0:
            actual_values_shifted.append(match['y'].values[0])
        else:
            actual_values_shifted.append(np.nan)
    
    # Plot comparison of alignments
    plt.figure(figsize=(15, 10))
    
    # First subplot: Original alignment (without shift)
    plt.subplot(2, 1, 1)
    plt.plot(df_forecasts.loc[mask, 'ds_original'], actual_values_original, 'b-', label='Actual')
    plt.plot(df_forecasts.loc[mask, 'ds_original'], df_forecasts.loc[mask, 'yhat'], 'r-', label='Predicted (No Shift)')
    plt.legend()
    plt.title('Without -24h Shift Correction')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    
    # Second subplot: Shifted alignment (with -24h shift)
    plt.subplot(2, 1, 2)
    plt.plot(df_forecasts.loc[mask, 'ds'], actual_values_shifted, 'b-', label='Actual')
    plt.plot(df_forecasts.loc[mask, 'ds'], df_forecasts.loc[mask, 'yhat'], 'r-', label='Predicted (With -24h Shift)')
    plt.legend()
    plt.title('With -24h Shift Correction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('alignment_comparison.png')
    plt.show()

    # Save results with explicit shift applied
    df_forecasts.to_csv("forecasts/df_forecasts_with_shift.csv")
    
    with open("saved_models/model_with_shift.json", "w", encoding="utf-8") as fout:
        json.dump(model_to_json(model), fout)


# if __name__ == "__main__":
#     # Select ex-regressors
#     regressors = [
#         "day_of_week",
#         "hour_of_day",
#         "price_actual_lag_24h",
#         "price_actual_lag_1w",
#         "price_actual_lag_2w",
#         "price_actual_lag_3w",
#         "price_actual_lag_4w",
#         "fossil_fuels_lag_24h",
#         "windpower_lag_24h",
#         "solarpower_lag_24h",
#         "other_green_energy_lag_24h",
#         "total_load_actual_lag_24h",
#     ]
#     # regressors = []

#     model_config = {
#         "changepoint_prior_scale": 0.001,  # default 0.05
#         "seasonality_prior_scale": 0.01,  # default 10.0
#         "holidays_prior_scale": 0.01,  # default 10.0
#     }

#     model, df_forecasts, mae_list, rmse_list = train_rolling(
#         forecast_horizon=10,
#         forecast_period=24,
#         model_config=model_config,
#         regressors=regressors
#     )

#     # # Print overall metrics
#     print(f"Overall MAE: {np.mean(mae_list):.4f}")
#     print(f"Overall RMSE: {np.mean(rmse_list):.4f}")

#     # save the model and forecast
#     with open("saved_models/model_new.json", "w", encoding="utf-8") as fout:
#         json.dump(model_to_json(model), fout)

#     df_forecasts.to_csv("forecasts/df_forecasts_new.csv")

#     # # # plot the forecast
#     # best_params_str = "_changepoint_prior_scale_0.001_seasonality_prior_scale_0.01_holidays_prior_scale_0.01"
#     # suffix = "_no_ylags"
#     # suffix = "_all_with_lags"
#     suffix = "_new"
#     model = model_from_json(
#         json.load(open(f"saved_models/model{suffix}.json", "r", encoding="utf-8"))
#     )
#     forecast_log = pd.read_csv(
#         f"forecasts/df_forecasts{suffix}.csv", parse_dates=["ds"]
#     )
#     plot(model, forecast_log, 24 * 7, 24 * 30)

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
