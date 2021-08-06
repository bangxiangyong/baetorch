import os
import time

from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.evaluation import calc_auroc, calc_avgprc
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_vi import BAE_VI

from baetorch.baetorch.models_v2.base_autoencoder import BAE_BaseClass
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
import numpy as np
import timeit

from baetorch.baetorch.util.sghmc import SGHMC
from baetorch.baetorch.util.truncated_gaussian import TruncatedNormal

random_seed = 999
bae_set_seed(random_seed)

train_size = 0.8

# === PREPARE DATA ===
base_folder = "F:\\understanding-bae\\uncertainty_ood\\od_benchmark"
mat_file_list = os.listdir(base_folder)
mat_file = mat_file_list[0]  # hard ones = 0, 2, 10, -3
mat = loadmat(os.path.join(base_folder, mat_file))
X = mat["X"]
y = mat["y"].ravel()

# get outliers and inliers
x_outliers, x_inliers = get_outliers_inliers(X, y)

x_outliers_train = x_outliers.copy()
x_outliers_test = x_outliers.copy()

x_inliers_train, x_inliers_test = train_test_split(
    x_inliers, train_size=train_size, shuffle=True, random_state=random_seed
)
x_inliers_train, x_inliers_valid = train_test_split(
    x_inliers_train, train_size=train_size, shuffle=True, random_state=random_seed
)

# === FIT DATA ===
skip = True
# skip = False
use_cuda = True
twin_output = False
# twin_output = True
homoscedestic_mode = "every"
# homoscedestic_mode = "single"
# homoscedestic_mode = "none"
clip_data_01 = True
# likelihood = "gaussian"
# likelihood = "laplace"
likelihood = "truncated_gaussian"
weight_decay = 0.0000000001
# weight_decay = 0.01
# weight_decay = 0.000001
# weight_decay = 0.000
# anchored = False
anchored = True
# sparse_scale = 0.0000001
sparse_scale = 0.00
n_stochastic_samples = 100
# n_ensemble = 5
n_ensemble = 1

# scaling
scaler = MinMaxScaler().fit(x_inliers_train)
x_inliers_train = scaler.transform(x_inliers_train)
x_inliers_valid = scaler.transform(x_inliers_valid)
x_inliers_test = scaler.transform(x_inliers_test)
x_outliers_test = scaler.transform(x_outliers_test)
x_outliers_train = scaler.transform(x_outliers_train)

if clip_data_01:
    x_inliers_train = np.clip(x_inliers_train, 0, 1)
    x_inliers_test = np.clip(x_inliers_test, 0, 1)
    x_outliers_test = np.clip(x_outliers_test, 0, 1)
    x_inliers_valid = np.clip(x_inliers_valid, 0, 1)
    x_outliers_train = np.clip(x_outliers_train, 0, 1)


input_dim = x_inliers_train.shape[1]
chain_params = [
    {
        "base": "linear",
        # "architecture": [input_dim, input_dim * 4, input_dim * 4, input_dim * 4],
        "architecture": [input_dim, input_dim * 2, input_dim * 4, input_dim * 5],
        # "architecture": [input_dim, input_dim * 2, input_dim * 4, input_dim // 5],
        "activation": "selu",
        "norm": True,
        # "bias":True
    }
]

# lin_autoencoder = BAE_MCDropout(
#     chain_params=chain_params,
#     last_activation="sigmoid",
#     last_norm=True,
#     twin_output=twin_output,
#     # twin_params={"activation": "leakyrelu", "norm": True},
#     twin_params={"activation": "softplus", "norm": False},
#     skip=skip,
#     use_cuda=use_cuda,
#     scaler_enabled=False,
#     homoscedestic_mode=homoscedestic_mode,
#     likelihood=likelihood,
#     weight_decay=weight_decay,
#     anchored=anchored,
#     sparse_scale=sparse_scale,
#     num_test_samples=n_stochastic_samples,
# )

#
# lin_autoencoder = BAE_VI(
#     chain_params=chain_params,
#     last_activation="sigmoid",
#     last_norm=True,
#     twin_output=twin_output,
#     # twin_params={"activation": "leakyrelu", "norm": True},
#     twin_params={"activation": "softplus", "norm": False},
#     skip=skip,
#     use_cuda=use_cuda,
#     scaler_enabled=False,
#     homoscedestic_mode=homoscedestic_mode,
#     likelihood=likelihood,
#     weight_decay=weight_decay,
#     num_test_samples=n_stochastic_samples,
# )

lin_autoencoder = BAE_Ensemble(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=twin_output,
    # twin_params={"activation": "none", "norm": False},
    twin_params={"activation": "leakyrelu", "norm": False},  # truncated gaussian
    # twin_params={"activation": "leakyrelu", "norm": True},  # truncated gaussian
    # twin_params={"activation": "softplus", "norm": False},
    # twin_params={"activation": "softplus", "norm": True},
    skip=skip,
    use_cuda=use_cuda,
    scaler_enabled=False,
    homoscedestic_mode=homoscedestic_mode,
    likelihood=likelihood,
    weight_decay=weight_decay,
    num_samples=n_ensemble,
    sparse_scale=sparse_scale,
    anchored=anchored,
)

# run lr_range_finder
temp_dataloader = convert_dataloader(
    x_inliers_train,
    batch_size=len(x_inliers_train) // 4,
)

# temp_dataloader = convert_dataloader(
#     x_inliers_train, batch_size=len(x_inliers_train) // 3, drop_last=True
# )

# Min lr:1.1e-06 , Max lr: 0.00234
# half_iterations = np.clip(len(x_inliers_train) // 2, 1, np.inf)
# lin_autoencoder.init_scheduler(
#     half_iterations=half_iterations, min_lr=1.1e-06, max_lr=0.0234
# )

# min_lr, max_lr, half_iter = run_auto_lr_range_v4(
#     temp_dataloader,
#     lin_autoencoder,
#     window_size=1,
#     num_epochs=10,
#     run_full=True,
# )

# # # start fitting
# for i in range(15):
#     lin_autoencoder.fit(temp_dataloader, num_epochs=20)

# time_method(lin_autoencoder.fit, temp_dataloader, num_epochs=100)
#
# for i in range(10):
#     lin_autoencoder.fit(temp_dataloader, num_epochs=5)


sghmc_optim = SGHMC(
    lin_autoencoder.autoencoder.parameters(),
    lr=lin_autoencoder.learning_rate,
    num_burn_in_steps=10,
)


#
# # start predicting
# nll_key = "nll"
#
# # ae_inliers_pred = lin_autoencoder.predict(
# #     x_inliers_test, select_keys=[nll_key], num_test_samples=n_stochastic_samples
# # )
# # ae_outliers_pred = lin_autoencoder.predict(
# #     x_outliers_test, select_keys=[nll_key], num_test_samples=n_stochastic_samples
# # )
#
# ae_inliers_pred = lin_autoencoder.predict(x_inliers_test, select_keys=[nll_key])
# ae_outliers_pred = lin_autoencoder.predict(x_outliers_test, select_keys=[nll_key])
#
#
# # evaluate AUROC and AVGPRC
# auroc_ood = calc_auroc(
#     ae_inliers_pred[nll_key].mean(-1), ae_outliers_pred[nll_key].mean(-1)
# )
# avgprc_ood = calc_avgprc(
#     ae_inliers_pred[nll_key].mean(-1), ae_outliers_pred[nll_key].mean(-1)
# )
#
# print("AUROC : {:.5f}".format(auroc_ood))
# print("AVG-PRC : {:.5f}".format(avgprc_ood))
#
#
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(lin_autoencoder.losses)


# ===========convert nll to gaussian=======

# import torch
#
# ae_inliers_pred = lin_autoencoder.predict(
#     x_inliers_test, select_keys=["y_mu", "y_sigma", nll_key]
# )
# ae_outliers_pred = lin_autoencoder.predict(
#     x_outliers_test, select_keys=["y_mu", "y_sigma", nll_key]
# )
#
# tc_inlier = (
#     TruncatedNormal(
#         loc=torch.from_numpy(ae_inliers_pred["y_mu"]).float(),
#         scale=torch.from_numpy(np.sqrt(np.exp(ae_inliers_pred["y_sigma"]))).float(),
#         a=0.0,
#         b=1.0,
#     )
#     .log_prob(torch.from_numpy(x_inliers_test).float())
#     .detach()
#     .cpu()
#     .numpy()
# ).mean(-1)
#
# tc_outlier = (
#     TruncatedNormal(
#         loc=torch.from_numpy(ae_outliers_pred["y_mu"]).float(),
#         scale=torch.from_numpy(np.sqrt(np.exp(ae_outliers_pred["y_sigma"]))).float(),
#         a=0.0,
#         b=1.0,
#     )
#     .log_prob(torch.from_numpy(x_outliers_test).float())
#     .detach()
#     .cpu()
#     .numpy()
# ).mean(-1)
#
#
# tc_auroc_ood = calc_auroc(-tc_inlier, -tc_outlier)
# print(tc_auroc_ood)
