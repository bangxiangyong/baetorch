import copy

import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn import Flatten

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v3, run_auto_lr_range_v4
from baetorch.baetorch.models_v2.base_autoencoder import (
    create_linear_chain,
    create_conv_chain,
    get_conv_latent_shapes,
    TwinOutputModule,
    AutoencoderModule,
    BAE_BaseClass,
)
from baetorch.baetorch.models_v2.base_layer import Reshape
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed

bae_set_seed(123)

# === LINEAR AE ===
dummy_data = torch.randn((100, 32))
input_dim = dummy_data.shape[1]
chain_params = [
    {
        "base": "linear",
        "architecture": [input_dim, input_dim * 2, input_dim * 4],
        "activation": "selu",
        "norm": True,
    }
]

lin_autoencoder = BAE_BaseClass(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=True,
    twin_params={"activation": "none", "norm": False},
    skip=True,
    use_cuda=True,
    scaler_enabled=True,
)

# fit-predict model
lin_autoencoder.fit(dummy_data, num_epochs=10)
ae_pred = lin_autoencoder.predict(dummy_data)

# run lr_range_finder
dummy_dataloader = convert_dataloader(
    dummy_data,
    batch_size=len(dummy_data) // 2,
)

min_lr, max_lr, half_iter = run_auto_lr_range_v4(
    dummy_dataloader,
    lin_autoencoder,
    window_size=3,
    num_epochs=10,
    run_full=False,
)

lin_autoencoder.fit(dummy_dataloader, num_epochs=10)
