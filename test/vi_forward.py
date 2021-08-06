from baetorch.baetorch.models_v2.vi_layer import VI_AutoencoderModule
import torch

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
        "activation": "leakyrelu",
        "norm": True,
    },
    {
        "base": "linear",
        "architecture": [100, latent_dim],
        "activation": "leakyrelu",
        "norm": True,
    },
]
vi_autoencoder = VI_AutoencoderModule(
    chain_params=chain_params,
    skip=False,
    twin_output=False,
)
vi_skip_autoencoder = VI_AutoencoderModule(
    chain_params=chain_params,
    skip=True,
    twin_output=False,
)
vi_skip_twin_autoencoder = VI_AutoencoderModule(
    chain_params=chain_params,
    skip=False,
    twin_output=True,
    twin_params={"activation": "none"},
)

chain_params = [
    {
        "base": "conv2d",
        "input_dim": input_dim,
        "conv_channels": [input_channel, 10, 25],
        "conv_stride": [2, 1],
        "conv_kernel": [2, 1],
        "activation": "none",
        "norm": True,
    }
]

conv_vi_autoencoder = VI_AutoencoderModule(
    chain_params=chain_params,
    skip=True,
    twin_output=True,
)

y_pred, kl_loss = vi_autoencoder(dummy_data)
y_pred_skip, kl_loss = vi_skip_autoencoder(dummy_data)
y_pred_skip_twin, kl_loss = vi_skip_twin_autoencoder(dummy_data)
y_pred_conv, kl_loss = conv_vi_autoencoder(dummy_data)

if torch.cuda.is_available():
    dummy_data = dummy_data.cuda()
    vi_autoencoder.set_cuda(True)
    y_pred = vi_autoencoder(dummy_data)
