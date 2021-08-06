import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from baetorch.baetorch.evaluation import calc_auroc, calc_avgprc
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.util.invert import Invert
from baetorch.baetorch.util.misc import time_method
from baetorch.baetorch.util.seed import bae_set_seed

random_seed = 999

bae_set_seed(random_seed)
# train_set_name = "SVHN"
# train_set_name = "FashionMNIST"
# train_set_name = "MNIST"
# train_set_name = "CELEBA"
train_set_name = "CIFAR"


# ==============PREPARE DATA==========================
shuffle = True
# data_transform = transforms.Compose(
#     [transforms.Resize((28, 28)), transforms.ToTensor()]
# )
data_transform = transforms.Compose([transforms.ToTensor()])
train_batch_size = 100
test_samples = 100

train_celeba_dataset = ImageFolder(
    root="F:\\understanding-bae\\baetorch\\test\\data-celeba\\celeba\\img_align_celeba",
    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
)
test_celeba_dataset = ImageFolder(
    root="F:\\understanding-bae\\baetorch\\test\\data-celeba\\celeba\\img_test_celeba",
    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
)

# celeba_dataset = datasets.CelebA(
#     root="E:\\PhD Dataset\\celeba\\",
#     transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
#     download=False,
# )
# train_set, val_set = torch.utils.data.random_split(celeba_dataset, [50000, 10000])

if train_set_name == "CELEBA":
    train_loader = torch.utils.data.DataLoader(
        train_celeba_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle,
    )

    test_loader = torch.utils.data.DataLoader(
        test_celeba_dataset,
        batch_size=test_samples,
        shuffle=True,
    )
    ood_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data-cifar10",
            train=False,
            download=True,
            transform=data_transform,
        ),
        batch_size=test_samples,
        shuffle=True,
    )
if train_set_name == "CIFAR":
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data-cifar", train=True, download=True, transform=data_transform
        ),
        batch_size=train_batch_size,
        shuffle=shuffle,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data-cifar", train=False, download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=False,
    )

    ood_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "data-svhn", split="test", download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=False,
    )

    # ood_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR100(
    #         "data-cifar100",
    #         download=True,
    #         transform=data_transform,
    #     ),
    #     batch_size=test_samples,
    #     shuffle=False,
    # )

    # ood_loader = torch.utils.data.DataLoader(
    #     datasets.Caltech101(
    #         "data-caltech101",
    #         download=True,
    #         transform=transforms.Compose(
    #             [
    #                 # transforms.ToPILImage(),
    #                 transforms.Lambda(lambda image: image.convert("RGB")),
    #                 transforms.Resize((32, 32)),
    #                 transforms.ToTensor(),
    #             ]
    #         ),
    #     ),
    #     batch_size=test_samples,
    #     shuffle=False,
    # )

    # ood_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(
    #         "data-fashion-mnist",
    #         train=False,
    #         download=True,
    #         transform=transforms.Compose(
    #             [
    #                 transforms.Grayscale(num_output_channels=3),
    #                 Invert(),
    #                 transforms.ToTensor(),
    #             ]
    #         ),
    #     ),
    #     batch_size=test_samples,
    #     shuffle=False,
    # )

elif train_set_name == "SVHN":
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "data-svhn", split="train", download=True, transform=data_transform
        ),
        batch_size=train_batch_size,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "data-svhn", split="test", download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=False,
    )
    ood_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data-cifar", train=False, download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=False,
    )
elif train_set_name == "FashionMNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "data-fashion-mnist", train=True, download=True, transform=data_transform
        ),
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "data-fashion-mnist", train=False, download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=False,
    )
    ood_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data-mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=test_samples,
        shuffle=False,
    )

elif train_set_name == "MNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data-mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=train_batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data-mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=test_samples,
        shuffle=False,
    )

    # ood_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(
    #         "data-fashion-mnist", train=False, download=True, transform=data_transform
    #     ),
    #     batch_size=test_samples,
    #     shuffle=False,
    # )

    ood_loader = torch.utils.data.DataLoader(
        datasets.Omniglot(
            "data-omniglot",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((28, 28)),
                    Invert(),
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=test_samples,
        shuffle=False,
    )

# ================BAE PARAMETERS====================

skip = True
# skip = False
use_cuda = True
twin_output = False
# twin_output = True
# homoscedestic_mode = "every"
# homoscedestic_mode = "single"
homoscedestic_mode = "none"
clip_data_01 = True
# likelihood = "ssim"
# likelihood = "gaussian"
likelihood = "laplace"
# likelihood = "bernoulli"
# likelihood = "cbernoulli"
# likelihood = "truncated_gaussian"
weight_decay = 0.0000000001
# weight_decay = 0.01
# weight_decay = 0.000001
anchored = False
# anchored = True
# sparse_scale = 0.0000001
sparse_scale = 0.00
n_stochastic_samples = 100
# n_ensemble = 5
n_ensemble = 1

if train_set_name == "CIFAR" or train_set_name == "SVHN" or train_set_name == "CELEBA":
    input_dim = list([32, 32])
    input_channel = 3
else:
    input_dim = list([28, 28])
    input_channel = 1

latent_dim = 1000
chain_params = [
    # {
    #     "base": "conv2d",
    #     "input_dim": input_dim,
    #     "conv_channels": [input_channel, 30, 64, 80],
    #     "conv_stride": [2, 1, 2],
    #     "conv_kernel": [8, 4, 2],
    #     "activation": "selu",
    #     "norm": True,
    #     "se_block": True,
    # },
    # {
    #     "base": "conv2d",
    #     "input_dim": input_dim,
    #     "conv_channels": [input_channel, 30, 64,124],
    #     "conv_stride": [2, 1,2],
    #     "conv_kernel": [8, 4,4],
    #     "activation": "leakyrelu",
    #     "norm": True,
    #     "se_block": False,
    # },
    {
        "base": "conv2d",
        "input_dim": input_dim,
        "conv_channels": [input_channel, 32, 64],
        "conv_stride": [2, 1],
        "conv_kernel": [4, 2],
        "activation": "leakyrelu",
        "norm": True,
        "se_block": True,
    },
    # {
    #     "base": "conv2d",
    #     "input_dim": input_dim,
    #     "conv_channels": [input_channel, 30, 64],
    #     "conv_stride": [2, 1],
    #     "conv_kernel": [8, 4],
    #     "activation": "silu",
    #     "norm": True,
    #     "se_block": False,
    # },
    {
        "base": "linear",
        "architecture": [latent_dim],
        "activation": "leakyrelu",
        "norm": True,
    },
]

bae_model = BAE_Ensemble(
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

min_lr, max_lr, half_iter = run_auto_lr_range_v4(
    train_loader,
    bae_model,
    window_size=1,
    num_epochs=10,
    run_full=False,
)

if isinstance(bae_model, BAE_SGHMC):
    bae_model.fit(train_loader, burn_epoch=10, sghmc_epoch=5)
    # bae_model.fit(train_loader, burn_epoch=10, sghmc_epoch=5)
    # bae_model.fit(train_loader, burn_epoch=3, sghmc_epoch=3, save_every=1)
else:
    time_method(bae_model.fit, train_loader, num_epochs=10)
    # time_method(bae_model.fit, train_loader, num_epochs=3)

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

print("AUROC : {:.5f}".format(auroc_ood))
print("AVG-PRC : {:.5f}".format(avgprc_ood))

# =================================
import matplotlib.pyplot as plt
import numpy as np

in_data = next(iter(test_loader))[0][:2].detach().cpu().numpy()
out_data = next(iter(ood_loader))[0][:2].detach().cpu().numpy()
ae_inliers_sample = bae_model.predict(in_data, select_keys=[nll_key, "y_mu"])
ae_outliers_sample = bae_model.predict(out_data, select_keys=[nll_key, "y_mu"])

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax1.imshow(np.moveaxis(in_data[0], 0, 2))
ax2.imshow(np.moveaxis(out_data[0], 0, 2))

ax3.imshow(np.moveaxis(ae_inliers_sample["y_mu"].mean(0)[0], 0, 2))
ax4.imshow(np.moveaxis(ae_outliers_sample["y_mu"].mean(0)[0], 0, 2))

ax5.imshow((ae_inliers_sample[nll_key].mean(0)[0].mean(0)))
ax6.imshow((ae_outliers_sample[nll_key].mean(0)[0].mean(0)))

ax1.set_title("INLIER: {:.4f}".format(ae_inliers_sample[nll_key].mean(0)[0].mean()))
ax2.set_title("OUTLIER: {:.4f}".format(ae_outliers_sample[nll_key].mean(0)[0].mean()))


plt.figure()
plt.boxplot(
    [
        flatten_np(ae_inliers_pred[nll_key].mean(0)).mean(-1),
        flatten_np(ae_outliers_pred[nll_key].mean(0)).mean(-1),
    ]
)

# flatten np
plt.figure()
plt.hist(
    flatten_np(ae_inliers_pred[nll_key].mean(0)).mean(-1), density=True, alpha=0.75
)
plt.hist(
    flatten_np(ae_outliers_pred[nll_key].mean(0)).mean(-1), density=True, alpha=0.75
)
