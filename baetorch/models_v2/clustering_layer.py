import torch
import torch.nn as nn
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder
from baetorch.baetorch.models.base_layer import DenseLayers
from baetorch.baetorch.util.seed import bae_set_seed

# Clustering layer definition (see DCEC article for equations)
class ClusteringLayer(nn.Module):
    def __init__(self, architecture=[], activation='tanh', last_activation='none', input_size=10, output_size=10, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha

        if len(architecture) != 0:
            if len(architecture) == 1:
                self.dense_decoder = DenseLayers(architecture=[],
                                                           output_size=architecture[-1],
                                                           input_size=self.input_size,
                                                           activation=activation,
                                                           last_activation=last_activation)
            else:
                self.dense_decoder = DenseLayers(architecture=architecture[:-1],
                                                           output_size=architecture[-1],
                                                           input_size=self.input_size,
                                                           activation=activation,
                                                           last_activation=last_activation)
            self.weight = nn.Parameter(torch.Tensor(self.output_size, architecture[-1]))
        else:
            self.dense_decoder = None
            self.weight = nn.Parameter(torch.Tensor(self.output_size, self.input_size))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        if self.dense_decoder is not None:
            x = self.dense_decoder(x)
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)
