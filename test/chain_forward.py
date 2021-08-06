from torch.nn import Flatten

from baetorch.baetorch.models_v2.base_autoencoder import (
    create_conv_chain,
    create_linear_chain,
    get_conv_latent_shapes,
)
import torch
import numpy as np

from baetorch.baetorch.models_v2.base_layer import Reshape


# === CREATE DUMMIES ===
data_2d = torch.randn((10, 3, 32, 32))
input_dim = data_2d.shape[-2:]
conv_channels = [data_2d.shape[1], 5, 8]
linear_latent_dims = [100]

# === SPAWN LAYERS ===
# spawn encoder convolutional chain
enc_conv2d_chain = torch.nn.Sequential(
    *create_conv_chain(
        input_dim=input_dim,
        conv_channels=conv_channels,
        conv_kernel=[2, 2],
        conv_stride=[2, 1],
        base="conv2d",
        activation="leakyrelu",
        norm=True,
        last_activation="sigmoid",
        last_norm=True,
    )
)

# spawn decoder convolutional chain (transpose=True)
dec_conv2d_chain = torch.nn.Sequential(
    *create_conv_chain(
        input_dim=input_dim,
        conv_channels=conv_channels,
        conv_kernel=[2, 2],
        conv_stride=[2, 1],
        base="conv2d",
        activation="leakyrelu",
        norm=True,
        last_activation="sigmoid",
        last_norm=True,
        transpose=True,
    )
)
twin_dec_conv2d_chain = torch.nn.Sequential(
    *create_conv_chain(
        input_dim=input_dim,
        conv_channels=conv_channels,
        conv_kernel=[2, 2],
        conv_stride=[2, 1],
        base="conv2d",
        activation="leakyrelu",
        norm=True,
        last_activation="sigmoid",
        last_norm=True,
        transpose=True,
        twin_output=True,
        twin_params={"activation": "none", "norm": False},
    )
)

# pass through to get shapes
latent_shapes, flatten_latent_shapes = get_conv_latent_shapes(
    enc_conv2d_chain, *input_dim
)

# spawn connector reshaping layers
enc_flatten = Flatten()
dec_reshape = Reshape(latent_shapes[-1])

# spawn linear layers
enc_linear = torch.nn.Sequential(
    *create_linear_chain(
        architecture=[flatten_latent_shapes[-1]] + linear_latent_dims,
        activation="leakyrelu",
        norm=True,
        bias=False,
        last_activation="sigmoid",
        last_norm=True,
        transpose=False,
    )
)

dec_linear = torch.nn.Sequential(
    *create_linear_chain(
        architecture=[flatten_latent_shapes[-1]] + linear_latent_dims,
        activation="leakyrelu",
        norm=True,
        bias=False,
        last_activation="sigmoid",
        last_norm=True,
        transpose=True,
    )
)


# === FORWARD PASS ===
# forward pass convolutions only
enc_conv2d_outp = enc_conv2d_chain(data_2d)
dec_conv2d_outp = dec_conv2d_chain(enc_conv2d_outp)

# enc-dec flatten and reshape
reshaped_conv2d = dec_reshape(enc_flatten(enc_conv2d_outp))

# full forward pass
full_pass = dec_conv2d_chain(
    dec_reshape(dec_linear(enc_linear(enc_flatten(enc_conv2d_outp))))
)

# full forward pass with twin outputs
full_twin_pass = twin_dec_conv2d_chain(
    dec_reshape(dec_linear(enc_linear(enc_flatten(enc_conv2d_outp))))
)


# === CREATE DUMMIES ===
dummy_data = torch.randn((10, 32))
input_dim = dummy_data.shape[-1]
architecture = [input_dim, input_dim * 2, input_dim * 3]
activation = "leakyrelu"
last_activation = "sigmoid"

# === CREATE LAYERS ===
enc_linear = torch.nn.Sequential(
    *create_linear_chain(
        architecture=architecture,
        activation=activation,
        norm=True,
        last_activation=activation,
        last_norm=True,
    )
)

dec_linear = torch.nn.Sequential(
    *create_linear_chain(
        architecture=architecture,
        activation=activation,
        norm=True,
        last_activation=last_activation,
        last_norm=True,
        transpose=True,
    )
)

dec_twin_linear = torch.nn.Sequential(
    *create_linear_chain(
        architecture=architecture,
        activation=activation,
        norm=True,
        last_activation=last_activation,
        last_norm=True,
        transpose=True,
        twin_output=True,
        twin_params={"activation": "none", "norm": False},
    )
)

# === FORWARD PASS ===

enc_data = enc_linear(dummy_data)
dec_data = dec_linear(enc_data)
dec_twin_data = dec_twin_linear(enc_data)
