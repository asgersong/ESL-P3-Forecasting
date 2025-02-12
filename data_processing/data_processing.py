# %%
import pandas as pd

ENERGY_DATASET_PATH = "../dataset/raw/energy_dataset.csv"
WEATHER_DATASET_PATH = "../dataset/raw/weather_features.csv"

energy_dataset = pd.read_csv(ENERGY_DATASET_PATH)
weather_dataset = pd.read_csv(WEATHER_DATASET_PATH)

# %% [markdown]
# ## Combination and Drop of Columns

# %%
weather = weather_dataset.drop(['temp_min', 'temp_max', 'pressure', 'humidity', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_3h', 'weather_id'], axis=1)
print(weather.describe())   


# %%
# Load the dataset
df = energy_dataset

# Combine fossil fuel-based generation
fossil_fuels = [
    "generation fossil brown coal/lignite", "generation fossil coal-derived gas",
    "generation fossil gas", "generation fossil hard coal", "generation fossil oil",
    "generation fossil oil shale", "generation fossil peat"
]
df["fossil fuels combined"] = df[fossil_fuels].sum(axis=1)

# Rename and aggregate renewable sources
df["windpower"] = df[["generation wind offshore", "generation wind onshore"]].sum(axis=1)
df["windpower forecast"] = df[["forecast wind offshore eday ahead", "forecast wind onshore day ahead"]].sum(axis=1)
df["solarpower"] = df["generation solar"]
df["solarpower forecast"] = df["forecast solar day ahead"]

# Aggregate other green energy sources (excluding wind and solar)
green_energy_sources = [
    "generation biomass", "generation geothermal", "generation hydro pumped storage aggregated",
    "generation hydro run-of-river and poundage", "generation hydro water reservoir",
    "generation marine", "generation other", "generation other renewable", "generation waste"
]
df["other green energy"] = df[green_energy_sources].sum(axis=1)

# Select relevant columns
columns_to_keep = [
    "time", "fossil fuels combined", "windpower", "windpower forecast", "solarpower", "solarpower forecast",
    "other green energy", "total load forecast", "total load actual", "price day ahead", "price actual"
]
df = df[columns_to_keep]

# Save the processed dataset
df.to_csv("../dataset/processed/simplified_dataset.csv", index=False)

# Display the first few rows
print(df.head())



# %% [markdown]
# ## Clean data
# There are columns with missing values, so we should interpolate them.

# %%
import numpy as np

# set zero values to NaN
df = pd.read_csv("simplified_dataset.csv")
df = df.replace(0.0, np.nan)

# count the number of missing values
print(df.isnull().sum())


# interpolate using pandas
df = df.interpolate()

df.to_csv("../dataset/processed/interpolated_dataset.csv", index=False)


# %% [markdown]
# ## Dataset split
# We will split the dataset into training, validation, and test datasets using a 80/10/10 ratio.

# %%
import os

df = pd.read_csv("../dataset/processed/interpolated_dataset.csv")

# Calculate lengths for train, validation, and test sets
train_len = int(len(df) * 0.8)
val_len = int(len(df) * 0.1)

# Split the dataset
train, val, test = df[:train_len], df[train_len:train_len + val_len], df[train_len + val_len:]

# Separate features and target
train_x, train_y = train.drop(["price actual"], axis=1), train["price actual"]
val_x, val_y = val.drop(["price actual"], axis=1), val["price actual"]
test_x, test_y = test.drop(["price actual"], axis=1), test["price actual"]

# Create directories if they do not exist
os.makedirs("../dataset/processed/train", exist_ok=True)
os.makedirs("../dataset/processed/val", exist_ok=True)
os.makedirs("../dataset/processed/test", exist_ok=True)

# Save the datasets
train_x.to_csv("../dataset/processed/train/train_x.csv", index=False)
train_y.to_csv("../dataset/processed/train/train_y.csv", index=False)
val_x.to_csv("../dataset/processed/val/val_x.csv", index=False)
val_y.to_csv("../dataset/processed/val/val_y.csv", index=False)
test_x.to_csv("../dataset/processed/test/test_x.csv", index=False)
test_y.to_csv("../dataset/processed/test/test_y.csv", index=False)



