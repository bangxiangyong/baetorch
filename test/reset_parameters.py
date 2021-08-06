from baetorch.baetorch.models_v2.base_autoencoder import AutoencoderModule

chain_params = [
    {
        "base": "conv1d",
        "input_dim": 32,
        "conv_channels": [3, 10, 25],
        "conv_stride": [2, 2],
        "conv_kernel": [10, 10],
        "activation": "silu",
        "norm": True,
    },
    {
        "base": "linear",
        "architecture": [100],
        "activation": "leakyrelu",
        "norm": True,
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
autoencoder.reset_parameters()
