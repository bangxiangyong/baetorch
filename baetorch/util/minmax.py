from sklearn.preprocessing import MinMaxScaler
import numpy as np
from baetorch.baetorch.models.base_layer import flatten_np

class MultiMinMaxScaler():
    def __init__(self, clip=True):
        self.scaler = MinMaxScaler()
        self.clip = clip

    def fit(self, x_train, y_train=None):
        if len(x_train.shape) > 2:
            self.scaler.fit(flatten_np(x_train))
        else:
            self.scaler.fit(x_train)

    def fit_transform(self, x_train, y_train=None):
        if len(x_train.shape) > 2:
            new_x_train = self.scaler.fit_transform(flatten_np(x_train)).reshape(x_train.shape)
        else:
            new_x_train = self.scaler.fit_transform(x_train)

        if self.clip:
            new_x_train = np.clip(new_x_train, 0, 1)

        return new_x_train

    def transform(self, x_test):
        if len(x_test.shape) > 2:
            new_x_test = self.scaler.transform(flatten_np(x_test)).reshape(x_test.shape)
        else:
            new_x_test = self.scaler.transform(x_test)

        if self.clip:
            new_x_test = np.clip(new_x_test, 0, 1)

        return new_x_test

    def inverse_transform(self, x_test):
        if len(x_test.shape) > 2:
            new_x_test = self.scaler.inverse_transform(flatten_np(x_test)).reshape(x_test.shape)
        else:
            new_x_test = self.scaler.inverse_transform(x_test)
        return new_x_test











