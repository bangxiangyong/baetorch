from baetorch.baetorch.models_v2.base_autoencoder import AutoencoderModule
import torch

# dropout layer is added to every block
# done by specifiying dropout in the chain param.

dummy_data = torch.randn((10, 3, 32)).float()
dropout_rate = 0.1
chain_params = [
    {
        "base": "conv1d",
        "input_dim": 32,
        "conv_channels": [3, 10, 25],
        "conv_stride": [2, 2],
        "conv_kernel": [10, 10],
        "activation": "silu",
        "norm": True,
        "dropout": dropout_rate,
    },
    {
        "base": "linear",
        "architecture": [100],
        "activation": "leakyrelu",
        "norm": True,
        "dropout": dropout_rate,
    },
]
autoencoder = AutoencoderModule(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=True,
    twin_params={"activation": "none"},
    skip=True,
)

y_cuda = autoencoder(dummy_data)
