import copy

import torch
from torch.nn import Flatten

from baetorch.baetorch.models_v2.base_autoencoder import (
    create_linear_chain,
    create_conv_chain,
    get_conv_latent_shapes,
    TwinOutputModule,
    AutoencoderModule,
)
from baetorch.baetorch.models_v2.base_layer import Reshape


# === LINEAR AE ===
dummy_data = torch.randn((10, 32))
input_dim = dummy_data.shape[1]
chain_params = [
    {
        "base": "linear",
        "architecture": [input_dim, input_dim * 2, input_dim * 4],
        "activation": "selu",
        "norm": True,
    }
]
lin_autoencoder = AutoencoderModule(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=False,
    twin_params={},
    skip=True,
)
y_pred = lin_autoencoder(dummy_data)

# === CONV-2D AE ===
dummy_data = torch.randn((10, 3, 32, 32))
input_dim = dummy_data.shape[2:]
input_channel = dummy_data.shape[1]
chain_params = [
    {
        "base": "conv2d",
        "input_dim": input_dim,
        "conv_channels": [input_channel, 10, 25],
        "conv_stride": [2, 1],
        "conv_kernel": [2, 1],
        "activation": "silu",
        "norm": True,
    }
]
conv_autoencoder = AutoencoderModule(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=False,
    twin_params={},
    skip=True,
)
y_pred = conv_autoencoder(dummy_data)

# === CONV-2D-LINEAR AE ===
dummy_data = torch.randn((10, 3, 32, 32))
input_dim = list(dummy_data.shape[2:])
input_channel = dummy_data.shape[1]
latent_dim = 50
chain_params = [
    {
        "base": "conv2d",
        "input_dim": input_dim,
        "conv_channels": [input_channel, 10, 25],
        "conv_stride": [2, 1],
        "conv_kernel": [2, 1],
        "activation": "silu",
        "norm": True,
    },
    {
        "base": "linear",
        "architecture": [100, latent_dim],
        "activation": "leakyrelu",
        "norm": True,
    },
]
conv_linear_autoencoder = AutoencoderModule(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=False,
    twin_params={},
    skip=True,
)
y_pred = conv_linear_autoencoder(dummy_data)

# === CONV-2D-LINEAR-TWIN OUTPUT-AE====
dummy_data = torch.randn((10, 3, 32, 35))
input_dim = list(dummy_data.shape[2:])
input_channel = dummy_data.shape[1]
chain_params = [
    {
        "base": "conv2d",
        "input_dim": input_dim,
        "conv_channels": [input_channel, 10, 25],
        "conv_stride": [2, 1],
        "conv_kernel": [2, 3],
        "activation": "silu",
        "norm": True,
    },
    {
        "base": "linear",
        "architecture": [latent_dim],
        "activation": "leakyrelu",
        "norm": True,
    },
]
twin_autoencoder = AutoencoderModule(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=True,
    twin_params={"activation": "none", "norm": False},
    skip=True,
)
y_pred = twin_autoencoder(dummy_data)

# === CONV 1D ===
dummy_data = torch.randn((10, 3, 250))
input_dim = dummy_data.shape[2]
input_channel = dummy_data.shape[1]
chain_params = [
    {
        "base": "conv1d",
        "input_dim": input_dim,
        "conv_channels": [input_channel, 10, 25],
        "conv_stride": [2, 2],
        "conv_kernel": [10, 10],
        "activation": "silu",
        "norm": True,
    }
]
conv1d_autoencoder = AutoencoderModule(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=False,
    twin_params={},
    skip=True,
)
y_pred = conv1d_autoencoder(dummy_data)

# === CONV-1D-LINEAR ===
dummy_data = torch.randn((10, 3, 250))
input_dim = dummy_data.shape[2]
input_channel = dummy_data.shape[1]
chain_params = [
    {
        "base": "conv1d",
        "input_dim": input_dim,
        "conv_channels": [input_channel, 10, 25],
        "conv_stride": [2, 2],
        "conv_kernel": [10, 10],
        "activation": "silu",
        "norm": True,
    },
    {
        "base": "linear",
        "architecture": [latent_dim],
        "activation": "leakyrelu",
        "norm": True,
    },
]
conv1d_lin_twin_autoencoder = AutoencoderModule(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=True,
    twin_params={"activation": "none"},
    skip=True,
)

y_pred = conv1d_lin_twin_autoencoder(dummy_data)
