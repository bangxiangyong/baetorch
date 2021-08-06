import torch

from baetorch.baetorch.models_v2.base_autoencoder import AutoencoderModule

if torch.cuda.is_available():
    dummy_data = torch.randn((10, 3, 32)).float()
    dummy_data = dummy_data.cuda()

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
    conv1d_lin_twin_autoencoder = AutoencoderModule(
        chain_params=chain_params,
        last_activation="sigmoid",
        last_norm=True,
        twin_output=True,
        twin_params={"activation": "none"},
        skip=True,
    )

    conv1d_lin_twin_autoencoder.set_cuda(True)
    y_cuda = conv1d_lin_twin_autoencoder(dummy_data)

    print(
        "ENCODER CUDA:"
        + str(next(conv1d_lin_twin_autoencoder.encoder.parameters()).is_cuda)
    )
    print(torch.cuda.get_device_name(0))
