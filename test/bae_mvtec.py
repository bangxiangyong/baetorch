from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision
import torch
from torchvision import datasets, transforms

from baetorch.baetorch.evaluation import calc_auroc, calc_avgprc
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.util.invert import ThresholdTransform, Invert
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed

# image_resize = (128, 128)
# image_resize = (64, 64)
image_resize = (32, 32)
image_transform = torchvision.transforms.Compose(
    [
        # torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize(image_resize),
        # Invert(),
        torchvision.transforms.ToTensor(),
        # ThresholdTransform(thr_255=125),
    ]
)
input_dim = list(image_resize)
input_channel = 3

# root_folder = "E:\\PhD Dataset\\mvtec_anomaly_detection\\mvtec_anomaly_detection.tar\\mvtec_anomaly_detection\\tile\\"
# root_folder = "E:\\PhD Dataset\\mvtec_anomaly_detection\\mvtec_anomaly_detection.tar\\mvtec_anomaly_detection\\grid\\"
# root_folder = "E:\\PhD Dataset\\mvtec_anomaly_detection\\mvtec_anomaly_detection.tar\\mvtec_anomaly_detection\\leather\\"
# root_folder = "E:\\PhD Dataset\\mvtec_anomaly_detection\\mvtec_anomaly_detection.tar\\mvtec_anomaly_detection\\grouped\\"
root_folder = "E:\\PhD Dataset\\mvtec_anomaly_detection\\mvtec_anomaly_detection.tar\\mvtec_anomaly_detection\\hazelnut\\"

# root_folder = "F:\\PhD Dataset\\grid"
root_train_folder = root_folder + "train"
train_dataset = ImageFolder(
    root=root_train_folder,
    transform=image_transform,
)

root_test_folder = root_folder + "test_good"
test_dataset = ImageFolder(
    root=root_test_folder,
    transform=image_transform,
)

root_ood_folder = root_folder + "test"
ood_dataset = ImageFolder(
    root=root_ood_folder,
    transform=image_transform,
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=True)

# ood_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10(
#         "data-cifar", train=False, download=True, transform=image_transform
#     ),
#     batch_size=batch_size,
#     shuffle=False,
# )
# ood_loader = torch.utils.data.DataLoader(
#     datasets.Omniglot("data-omniglot", download=True, transform=image_transform),
#     batch_size=batch_size,
#     shuffle=False,
# )

random_seed = 333
# random_seed = 11

bae_set_seed(random_seed)
# train_set_name = "SVHN"
train_set_name = "FashionMNIST"
# train_set_name = "MNIST"
# train_set_name = "CIFAR"

# ==============PREPARE DATA==========================
shuffle = True
data_transform = transforms.Compose([transforms.ToTensor()])
train_batch_size = 100
test_samples = 500

# ================BAE PARAMETERS====================

skip = True
# skip = False
use_cuda = True
# twin_output = False
twin_output = True
# homoscedestic_mode = "every"
homoscedestic_mode = "single"
# homoscedestic_mode = "none"
clip_data_01 = True
# likelihood = "ssim"
likelihood = "gaussian"
# likelihood = "laplace"
# likelihood = "bernoulli"
# likelihood = "cbernoulli"
# likelihood = "truncated_gaussian"
weight_decay = 0.0000000001
# weight_decay = 0.01
# weight_decay = 0.00000001
# weight_decay = 0.000
anchored = False
# anchored = True
sparse_scale = 0.0000001
# sparse_scale = 0.000001
# sparse_scale = 0.00
n_stochastic_samples = 10
# n_ensemble = 5
n_ensemble = 1

# if train_set_name == "CIFAR" or train_set_name == "SVHN":
#     input_dim = list([32, 32])
#     input_channel = 3
# else:
#     input_dim = list([28, 28])
#     input_channel = 1

latent_dim = 50
chain_params = [
    # {
    #     "base": "conv2d",
    #     "input_dim": input_dim,
    #     "conv_channels": [input_channel, 16, 32, 64],
    #     "conv_stride": [2, 2, 1],
    #     "conv_kernel": [10, 20, 5],
    #     "activation": "selu",
    #     "norm": True,
    # },
    {
        "base": "conv2d",
        "input_dim": input_dim,
        "conv_channels": [input_channel, 16, 32, 64],
        "conv_stride": [2, 1, 2],
        "conv_kernel": [4, 4, 2],
        "activation": "selu",
        "norm": True,
        "se_block": True,
        # "se_block": False,
    },
    {
        "base": "linear",
        "architecture": [latent_dim],
        "activation": "leakyrelu",
        "norm": True,
    },
    # {
    #     "base": "conv2d",
    #     "input_dim": input_dim,
    #     "conv_channels": [input_channel, 10, 12, 13, 14, 15, 6, 8, 9, 10, 11],
    #     "conv_stride": [2, 2, 1, 2, 1, 2, 1, 1, 1],
    #     "conv_kernel": [4, 4, 3, 4, 3, 4, 3, 3, 8],
    #     "activation": "selu",
    #     "norm": True,
    #     # "se_block": True,
    #     "se_block": False,
    # },
]

bae_model = BAE_Ensemble(
    chain_params=chain_params,
    last_activation="sigmoid",
    last_norm=True,
    twin_output=twin_output,
    # twin_params={"activation": "none", "norm": False},
    # twin_params={"activation": "leakyrelu", "norm": False},  # truncated gaussian
    # twin_params={"activation": "leakyrelu", "norm": True},  # truncated gaussian
    twin_params={"activation": "softplus", "norm": False},
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

# bae_model = BAE_MCDropout(
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
#     num_samples=n_stochastic_samples,
# )

# bae_model = BAE_SGHMC(
#     chain_params=chain_params,
#     last_activation="sigmoid",
#     last_norm=True,
#     twin_output=twin_output,
#     # twin_params={"activation": "none", "norm": False},
#     # twin_params={"activation": "leakyrelu", "norm": False},  # truncated gaussian
#     # twin_params={"activation": "leakyrelu", "norm": True},  # truncated gaussian
#     twin_params={"activation": "softplus", "norm": False}
#     if likelihood == "gaussian"
#     else {"activation": "leakyrelu", "norm": False},
#     # twin_params={"activation": "softplus", "norm": True},
#     skip=skip,
#     use_cuda=use_cuda,
#     scaler_enabled=False,
#     homoscedestic_mode=homoscedestic_mode,
#     likelihood=likelihood,
#     weight_decay=weight_decay,
#     num_samples=5,
#     sparse_scale=sparse_scale,
#     anchored=anchored,
# )

# min_lr, max_lr, half_iter = run_auto_lr_range_v4(
#     train_loader,
#     bae_model,
#     window_size=1,
#     num_epochs=10,
#     run_full=False,
# )

# bae_model.init_scheduler(
#     half_iterations=len(train_loader) // 2, min_lr=1.99e-05, max_lr=0.1
# )

if isinstance(bae_model, BAE_SGHMC):
    bae_model.fit(train_loader, burn_epoch=3, sghmc_epoch=2)
    # bae_model.fit(train_loader, burn_epoch=3, sghmc_epoch=3, save_every=1)
else:
    # time_method(bae_model.fit, train_loader, num_epochs=1)
    time_method(bae_model.fit, train_loader, num_epochs=3)
    # time_method(bae_model.fit, train_loader, num_epochs=1)
    # time_method(bae_model.fit, train_loader, num_epochs=2)
    # time_method(bae_model.fit, train_loader, num_epochs=10)
    # time_method(bae_model.fit, train_loader, num_epochs=2)

# nll_key = "se"
nll_key = "nll"

# ae_inliers_pred = bae_model.predict(next(iter(test_loader))[0], select_keys=[nll_key])
# ae_outliers_pred = bae_model.predict(next(iter(ood_loader))[0], select_keys=[nll_key])
ae_inliers_pred = bae_model.predict(test_loader, select_keys=[nll_key])
ae_outliers_pred = bae_model.predict(ood_loader, select_keys=[nll_key])


# evaluate AUROC and AVGPRC
auroc_ood = calc_auroc(
    flatten_np(ae_inliers_pred[nll_key].mean(0)).mean(-1),
    flatten_np(ae_outliers_pred[nll_key].mean(0)).mean(-1),
)

avgprc_ood = calc_avgprc(
    flatten_np(ae_inliers_pred[nll_key].mean(0)).mean(-1),
    flatten_np(ae_outliers_pred[nll_key].mean(0)).mean(-1),
)

# evaluate AUROC and AVGPRC.0
auroc_ood_var = calc_auroc(
    flatten_np(ae_inliers_pred[nll_key].var(0)).mean(-1),
    flatten_np(ae_outliers_pred[nll_key].var(0)).mean(-1),
)

avgprc_ood_var = calc_avgprc(
    flatten_np(ae_inliers_pred[nll_key].var(0)).mean(-1),
    flatten_np(ae_outliers_pred[nll_key].var(0)).mean(-1),
)

# evaluate AUROC and AVGPRC
auroc_ood_fvar = calc_auroc(
    flatten_np(ae_inliers_pred[nll_key].mean(0)).var(-1),
    flatten_np(ae_outliers_pred[nll_key].mean(0)).var(-1),
)

avgprc_ood_fvar = calc_avgprc(
    flatten_np(ae_inliers_pred[nll_key].mean(0)).var(-1),
    flatten_np(ae_outliers_pred[nll_key].mean(0)).var(-1),
)


print("AUROC : {:.5f}".format(auroc_ood))
print("AVG-PRC : {:.5f}".format(avgprc_ood))
print("AUROC : {:.5f}".format(auroc_ood_var))
print("AVG-PRC : {:.5f}".format(avgprc_ood_var))
print("FEATURE VAR-AUROC : {:.5f}".format(auroc_ood_fvar))
print("FEATURE VAR-AVG-PRC : {:.5f}".format(avgprc_ood_fvar))


# =================================
import matplotlib.pyplot as plt
import numpy as np

plot_key = "nll"
in_data = next(iter(test_loader))[0][:2].detach().cpu().numpy()
out_data = next(iter(ood_loader))[0][:2].detach().cpu().numpy()
ae_inliers_sample = bae_model.predict(in_data, select_keys=[plot_key, "y_mu"])
ae_outliers_sample = bae_model.predict(out_data, select_keys=[plot_key, "y_mu"])

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax1.imshow(np.moveaxis(in_data[0], 0, 2))
ax2.imshow(np.moveaxis(out_data[0], 0, 2))

ax3.imshow(np.moveaxis(ae_inliers_sample["y_mu"].mean(0)[0], 0, 2))
ax4.imshow(np.moveaxis(ae_outliers_sample["y_mu"].mean(0)[0], 0, 2))

ax5.imshow((ae_inliers_sample[plot_key].mean(0)[0].mean(0)))
ax6.imshow((ae_outliers_sample[plot_key].mean(0)[0].mean(0)))

ax1.set_title("INLIER")
ax2.set_title("OUTLIER")

plt.figure()
plt.boxplot(
    [
        flatten_np(ae_inliers_pred[nll_key].mean(0)).mean(-1),
        flatten_np(ae_outliers_pred[nll_key].mean(0)).mean(-1),
    ]
)
