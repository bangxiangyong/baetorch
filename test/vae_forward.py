import torch
from baetorch.baetorch.models_v2.vae import VAEModule


# LINEAR VAE
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
linear_vae = VAEModule(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=False,
    twin_params={},
    skip=True,
)

enc_mu, enc_sig = linear_vae.encoder(dummy_data)
y_sample, kl_loss = linear_vae(dummy_data)

# CONV2D VAE
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

conv_vae = VAEModule(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=False,
    twin_params={},
    skip=True,
)

enc_mu, enc_sig = conv_vae.encoder(dummy_data)
y_sample, kl_loss = conv_vae(dummy_data)

# === CONV-2D-LINEAR VAE ===
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
conv_linear_vae = VAEModule(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=True,
    twin_params={},
    skip=False,
    use_cuda=True,
)

y_sample, kl_loss = conv_linear_vae(dummy_data.cuda())

for param in list(conv_linear_vae.named_parameters()):
    print(param[0])
    print(param[1].is_cuda)
