from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ..models.base_layer import flatten_np
import torch


class MultiMinMaxScaler:
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
            new_x_train = self.scaler.fit_transform(flatten_np(x_train)).reshape(
                x_train.shape
            )
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
            new_x_test = self.scaler.inverse_transform(flatten_np(x_test)).reshape(
                x_test.shape
            )
        else:
            new_x_test = self.scaler.inverse_transform(x_test)
        return new_x_test


class TorchMinMaxScaler:
    def __init__(self, clip=True):
        self.min_np = None
        self.max_np = None
        self.fitted = False
        self.clip = clip

    @property
    def scale_np(self):
        return self.max_np - self.min_np

    @property
    def scale_torch(self):
        return self.max_torch - self.min_torch

    def get_shape(self, x):
        return list(x.shape[1:])

    def partial_fit(self, x):
        if self.min_np is not None:
            data_min = x.min(0)
            data_max = x.max(0)
            if isinstance(x, np.ndarray):
                self.min_np = np.minimum(self.min_np, data_min)
                self.max_np = np.maximum(self.max_np, data_max)
                self.min_torch = torch.from_numpy(self.min_np).float()
                self.max_torch = torch.from_numpy(self.max_np).float()
            else:
                self.min_torch = torch.minimum(self.min_torch, data_min[0])[0]
                self.max_torch = torch.maximum(self.max_torch, data_max[0])[0]
                self.min_np = self.min_torch.detach().cpu().numpy()
                self.max_np = self.max_torch.detach().cpu().numpy()
        else:
            self.fit(x)

    def fit(self, x):
        if len(x.shape) > 2:
            self.need_reshape = True
        else:
            self.need_reshape = False

        if isinstance(x, np.ndarray):
            self.min_np = x.min(0)
            self.max_np = x.max(0)
            self.min_torch = torch.from_numpy(self.min_np).float()
            self.max_torch = torch.from_numpy(self.max_np).float()

        elif isinstance(x, torch.Tensor):
            self.min_torch = x.min(0)[0]
            self.max_torch = x.max(0)[0]
            self.min_np = self.min_torch.detach().cpu().numpy()
            self.max_np = self.max_torch.detach().cpu().numpy()

        self.fitted = True

    def transform(self, x):
        if isinstance(x, np.ndarray):
            if self.clip:
                return np.clip((x - self.min_np) / self.scale_np, 0.0, 1.0)
            else:
                return (x - self.min_np) / self.scale_np
        elif isinstance(x, torch.Tensor):
            if self.clip:
                return (x - self.min_torch) / self.scale_torch
            else:
                return torch.clip((x - self.min_torch) / self.scale_torch, 0.0, 1.0)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
