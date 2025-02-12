"""Prophet prediction model for electricity price forecasting."""
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from prophet.plot import add_changepoints_to_plot


def load_and_prepare_data(filepath):
    """Load dataset and prepare it for Prophet modeling."""
    train = pd.read_csv(filepath)
    train = train.rename(columns={"time": "ds", "price actual": "y"})
    train["ds"] = pd.to_datetime(train["ds"], format="%Y-%m-%d %H:%M:%S%z", utc=True)
    train["ds"] = train["ds"].dt.tz_localize(None)  # Remove timezone
    return train[["ds", "y"]]


def train_model(train, changepoint_prior_scale=0.05):
    """Train a Prophet model on the given dataset."""
    model = Prophet(changepoint_prior_scale=changepoint_prior_scale)
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

    a = add_changepoints_to_plot(fig.gca(), model, forecast)

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


def main(saved_model=None):
    filepath = "../dataset/processed/train/train.csv"
    filepath_val = "../dataset/processed/val/val.csv"

    train_data = load_and_prepare_data(filepath)
    model = train_model(train_data) if not saved_model else load_model(saved_model)

    save_model(model, "prophet_model.json")

    forecast = make_predictions(model)
    plot_forecast(model, forecast, load_and_prepare_data(filepath_val))


if __name__ == "__main__":
    # main()
    main(saved_model="prophet_model.json")
