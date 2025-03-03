import os
import json
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_to_json, model_from_json
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.makedirs("saved_models", exist_ok=True)
os.makedirs("forecasts", exist_ok=True)

# Load data
df_train = pd.read_csv("../../dataset/processed/train/train.csv")
df_test = pd.read_csv("../../dataset/processed/test/test.csv")

df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)

# Rename and format columns for Prophet
df_train = df_train.rename(columns={"time": "ds", "price_actual": "y"})
df_test = df_test.rename(columns={"time": "ds", "price_actual": "y"})
df_train["ds"] = pd.to_datetime(df_train["ds"], format="%Y-%m-%d %H:%M:%S")
df_test["ds"] = pd.to_datetime(df_test["ds"], format="%Y-%m-%d %H:%M:%S")

# ensure that the data is sorted by time
df_train = df_train.sort_values("ds")
df_test = df_test.sort_values("ds")

df_train = df_train.drop(
    columns=[
        "price_day_ahead",
        "fossil_fuels",
        "windpower",
        "solarpower",
        "other_green_energy",
        "total_load_actual",
    ]
)

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


def create_model(m_conf, r, ex_regressor_prior_scale):
    """Create a Prophet model with the given configuration and regressors."""
    model_config_copy = m_conf.copy()
    model_config_copy.pop("ex_regressor_prior_scale", None)
    m = Prophet(**model_config_copy)
    m.add_country_holidays(country_name="ES")  # Add Spanish holidays
    for regressor in r:
        m.add_regressor(regressor, prior_scale=ex_regressor_prior_scale)
    return m


def train_model(m_conf=None, r=None, df=df_train):
    """Train a Prophet model with the given configuration and regressors."""
    if m_conf is None:
        m_conf = {}
    if r is None:
        r = []
    m = create_model(m_conf, r, m_conf["ex_regressor_prior_scale"])
    m.fit(df)
    return m


def plot_forecasts(m: Prophet, f_cv, num_plots=6):
    """
    Plots the model's actual values and forecasts.
    """
    _, ax = plt.subplots(figsize=(10, 6))
    h = m.history.copy()

    # Plot actual values
    ax.plot(h["ds"], h["y"], "-b", label="Actual")

    # Plot consecutive forecasts
    unique_cutoffs = f_cv["cutoff"].unique()
    for fc in unique_cutoffs[1:]:  # Skip first cutoff
        df_cv = f_cv[f_cv["cutoff"] == fc]
        ax.plot(df_cv["ds"], df_cv["yhat"], color="orange", linestyle="--")

    # add shared labels for forecasts
    ax.plot(
        df_cv["ds"], df_cv["yhat"], color="orange", linestyle="--", label="Forecast"
    )

    ax.grid()
    ax.legend()

    # Calculate RMSE and MAE for overall forecast
    rmse = mean_squared_error(f_cv["y"], f_cv["yhat"]) ** 0.5
    mae = mean_absolute_error(f_cv["y"], f_cv["yhat"])
    ax.set_title(f"Overall RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Plot randomly selected forecasts
    random_cutoffs = np.random.choice(unique_cutoffs, num_plots, replace=False)
    n_cols = 3
    n_rows = (num_plots + n_cols - 1) // n_cols
    _, axs = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    axs = axs.flatten()

    for i, fc in enumerate(random_cutoffs):
        df_cv = f_cv[f_cv["cutoff"] == fc]
        ax = axs[i]
        ax.plot(h["ds"], h["y"], "-b", label="Actual")
        ax.plot(
            df_cv["ds"], df_cv["yhat"], color="orange", linestyle="--", label="Forecast"
        )

        # Add uncertainty interval
        ax.fill_between(
            df_cv["ds"],
            df_cv["yhat_lower"],
            df_cv["yhat_upper"],
            color="gray",
            alpha=0.2,
            label="Uncertainty",
        )

        ax.set_xlim(
            df_cv["ds"].min() - pd.Timedelta(hours=12),
            df_cv["ds"].max() + pd.Timedelta(hours=12),
        )

        # rotate x-axis labels
        ax.tick_params(axis="x", rotation=45)

        ax.grid()
        ax.legend()

        # Calculate RMSE and MAE for this forecast
        y_true = h.set_index("ds").loc[df_cv["ds"]].reset_index()["y"].values
        y_pred = df_cv["yhat"].values
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        ax.set_title(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    SUFFIX = "_final"
    TRAIN = False

    if TRAIN:
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

        model_config = {
            "changepoint_prior_scale": 0.5,  # default 0.05
            "seasonality_prior_scale": 1.0,  # default 10.0
            "holidays_prior_scale": 0.001,  # default 10.0
            "ex_regressor_prior_scale": 1.0,  # default None
        }

        model = train_model(model_config, regressors)
        forecasts_cv = cross_validation(
            model,
            initial="1000 days",
            period="4 days",
            horizon="24 hours",
            parallel="processes",
        )
        df_p = performance_metrics(forecasts_cv)

        # save the model and forecasts_cv and df_p
        with open(f"saved_models/model{SUFFIX}.json", "w", encoding="utf-8") as fout:
            json.dump(model_to_json(model), fout)

        forecasts_cv.to_csv(f"forecasts/f_cv{SUFFIX}.csv")
        df_p.to_csv(f"forecasts/df_p{SUFFIX}.csv")

    # open the model and f_cv and df_p
    model = model_from_json(
        json.load(open(f"saved_models/model{SUFFIX}.json", "r", encoding="utf-8"))
    )
    forecasts_cv = pd.read_csv(f"forecasts/f_cv{SUFFIX}.csv", parse_dates=["ds"])
    df_p = pd.read_csv(f"forecasts/df_p{SUFFIX}.csv")

    plot_forecasts(model, forecasts_cv, 9)

    ######## HYPERPARAMETER TUNING ########
    # import itertools
    # import tqdm

    # param_grid = {
    #     "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
    #     "seasonality_prior_scale": [0.001, 0.01, 0.1, 1.0],
    #     "holidays_prior_scale": [0.001, 0.01, 0.1, 1.0],
    # }

    # ex_regressor_prior_scale = [0.001, 0.01, 0.1, 0.5, 1.0]

    # # Generate all combinations of parameters including ex_regressor_prior_scale
    # all_params = [
    #     dict(zip(list(param_grid.keys()) + ["ex_regressor_prior_scale"], v))
    #     for v in itertools.product(*param_grid.values(), ex_regressor_prior_scale)
    # ]
    # rmses = []  # Store the RMSEs for each params here
    # maes = []  # Store the MAEs for each params

    # # Use cross validation to evaluate all parameters
    # for params in tqdm.tqdm(all_params, desc="Tuning parameters"):
    #     print(f"Training model with params: {params}")
    #     m = train_model(
    #         params, regressors, params["ex_regressor_prior_scale"]
    #     )  # Fit model with given params
    #     df_cv = cross_validation(
    #         m,
    #         initial="1000 days",
    #         period="4 days",
    #         horizon="24 hours",
    #         parallel="processes",
    #     )
    #     df_p = performance_metrics(df_cv, rolling_window=1)

    #     y_true = df_cv["y"]
    #     y_pred = df_cv["yhat"]

    #     rmses.append(mean_squared_error(y_true, y_pred) ** 0.5)
    #     maes.append(mean_absolute_error(y_true, y_pred))

    # # Find the best parameters
    # tuning_results = pd.DataFrame(all_params)
    # tuning_results["rmse"] = rmses
    # tuning_results["mae"] = maes
    # print(tuning_results)

    # tuning_results.to_csv("tuning_results.csv")

    # best_params = all_params[np.argmin(rmses)]
    # print(f'\n\n Best params: {best_params}')

    ## RESULT: Best params: {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'holidays_prior_scale': 0.001, 'ex_regressor_prior_scale': 1.0}
