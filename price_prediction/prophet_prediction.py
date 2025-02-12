import pandas as pd
from prophet import Prophet

from dataset.paths import TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, VAL_X, VAL_Y

# train_x_df = pd.read_csv(TRAIN_X)