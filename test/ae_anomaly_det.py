import os
import time

from pyod.utils.data import get_outliers_inliers
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.evaluation import calc_auroc, calc_avgprc
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.base_autoencoder import BAE_BaseClass
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed
import numpy as np
import timeit

random_seed = 1
bae_set_seed(random_seed)

train_size = 0.8

# === PREPARE DATA ===
base_folder = "F:\\understanding-bae\\uncertainty_ood\\od_benchmark"
mat_file_list = os.listdir(base_folder)
mat_file = mat_file_list[10]  # hard ones = 0, 2, 10, -3
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
# twin_output = False
twin_output = True
homoscedestic_mode = "none"
clip_data_01 = True
likelihood = "gaussian"
weight_decay = 0.0000001
# weight_decay = 0.0000
anchored = True
# sparse_scale = 0.0000001
sparse_scale = 0.0

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
        "architecture": [input_dim, input_dim * 2, input_dim * 3, input_dim * 5],
        "activation": "selu",
        "norm": True,
        # "bias":True
    }
]

lin_autoencoder = BAE_BaseClass(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=twin_output,
    # twin_params={"activation": "leakyrelu", "norm": True},
    twin_params={"activation": "softplus", "norm": False},
    skip=skip,
    use_cuda=use_cuda,
    scaler_enabled=False,
    homoscedestic_mode=homoscedestic_mode,
    likelihood=likelihood,
    weight_decay=weight_decay,
    anchored=anchored,
    sparse_scale=sparse_scale,
)

# run lr_range_finder
temp_dataloader = convert_dataloader(
    x_inliers_train,
    batch_size=len(x_inliers_train) // 3,
)

# Min lr:1.1e-06 , Max lr: 0.00234
# half_iterations = np.clip(len(x_inliers_train) // 2, 1, np.inf)
# lin_autoencoder.init_scheduler(
#     half_iterations=half_iterations, min_lr=1.1e-06, max_lr=0.0234
# )

min_lr, max_lr, half_iter = run_auto_lr_range_v4(
    temp_dataloader,
    lin_autoencoder,
    window_size=1,
    num_epochs=10,
    run_full=True,
)

# # # start fitting
# for i in range(5):
#     lin_autoencoder.fit(temp_dataloader, num_epochs=50)

time_method(lin_autoencoder.fit, temp_dataloader, num_epochs=150)

# start predicting
nll_key = "nll"
ae_inliers_pred = lin_autoencoder.predict(x_inliers_test, select_keys=[nll_key])
ae_outliers_pred = lin_autoencoder.predict(x_outliers_test, select_keys=[nll_key])

# evaluate AUROC and AVGPRC
auroc_ood = calc_auroc(
    ae_inliers_pred[nll_key].mean(-1), ae_outliers_pred[nll_key].mean(-1)
)
avgprc_ood = calc_avgprc(
    ae_inliers_pred[nll_key].mean(-1), ae_outliers_pred[nll_key].mean(-1)
)

print("AUROC : {:.5f}".format(auroc_ood))
print("AVG-PRC : {:.5f}".format(avgprc_ood))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(lin_autoencoder.losses)
