from sklearn.preprocessing import MinMaxScaler
import numpy as np
from baetorch.baetorch.models.base_layer import flatten_np

class MultiMinMaxScaler():
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, x_train):
        self.scaler.fit(x_train)

    def fit_transform(self, x_train):
        if len(x_train.shape) > 2:
            x_train = self.scaler.fit_transform(flatten_np(x_train)).reshape(x_train.shape)
        else:
            x_train = self.scaler.fit_transform(x_train)
        return x_train

    def transform(self, x_test):
        if len(x_test.shape) > 2:
            x_test = np.clip(self.scaler.transform(flatten_np(x_test)).reshape(x_test.shape), 0, 1)
        else:
            x_test = self.scaler.transform(x_test)
        return x_test
