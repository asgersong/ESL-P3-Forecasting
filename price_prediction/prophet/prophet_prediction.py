"""Prophet prediction model for electricity price forecasting."""

from numpy import block
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from prophet.plot import add_changepoints_to_plot

import statsmodels.api as sm


def load_and_prepare_data(filepath):
    """Load dataset and prepare it for Prophet modeling."""
    train = pd.read_csv(filepath)
    train = train.rename(columns={"time": "ds", "price actual": "y"})
    train["ds"] = pd.to_datetime(train["ds"], format="%Y-%m-%d %H:%M:%S%z", utc=True)
    train["ds"] = train["ds"].dt.tz_localize(None)  # Remove timezone
    return train[["ds", "y"]]


def train_model(model_config, train):
    """Train a Prophet model on the given dataset."""
    model = Prophet(**model_config)
    model.fit(train)
    return model


def make_predictions(model: Prophet, periods=16):
    """Generate future predictions using the trained model."""
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


def plot_forecast(model: Prophet, forecast, real_data=None):
    """Plot the forecasted data."""
    fig = model.plot(forecast)

    if real_data is not None:
        plt.plot(real_data["ds"], real_data["y"], "r")

    # plt.xlim(forecast['ds'].max()-pd.Timedelta(weeks=5), forecast['ds'].max())

    _ = add_changepoints_to_plot(fig.gca(), model, forecast)

    fig.show()
    plt.show(block=True)


def save_model(model, filepath):
    """Save the trained model to a file."""
    with open(filepath, "w") as fout:
        fout.write(model_to_json(model))


def load_model(filepath):
    """Load a trained model from a file."""
    with open(filepath, "r") as fin:
        model = model_from_json(fin.read())
    return model

def plot_decomposition(df):
    """Plot the decomposed time series."""
    df = df.set_index('ds')
    decomposition = sm.tsa.seasonal_decompose(df['y'], model='additive')
    decomposition.plot()
    plt.show(block=True)

def main(model_config, run_config):
    filepath = run_config["filepath_train"]
    filepath_val = run_config["filepath_val"]
    saved_model = run_config["model_name"] + ".json" if run_config["load_model"] else None

    train_data = load_and_prepare_data(filepath)
    model = (
        train_model(model_config, train_data)
        if not saved_model
        else load_model(saved_model)
    )

    if run_config["save_model"]:
        save_model(model, run_config["model_name"] + ".json")

    forecast = make_predictions(model, run_config["num_periods"])
    if run_config["plot"]:
        plot_decomposition(train_data)
        plot_forecast(model, forecast, load_and_prepare_data(filepath_val))


if __name__ == "__main__":
    model_config = {
        "changepoint_prior_scale": 0.05,  # default 0.05
        "growth": "linear",  # default "linear"
        "n_changepoints": 100,  # default 25
        "weekly_seasonality": False,  # default auto
        "daily_seasonality": True,  # default auto
        "yearly_seasonality": True,  # default auto
    }

    run_config = {
        "filepath_train": "../../dataset/processed/train/train.csv",
        "filepath_val": "../../dataset/processed/val/val.csv",
        "load_model": True,
        "save_model": True,
        "model_name": "prophet",
        "plot": True,
        "num_periods": 168,
    }

    main(model_config, run_config)
