import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torchvision import datasets, transforms

from baetorch.baetorch.evaluation import calc_auroc
from baetorch.baetorch.models_v2.base_layer import Flatten
from baetorch.baetorch.util.seed import bae_set_seed
import numpy as np

random_seed = 3145

bae_set_seed(random_seed)
# train_set_name = "SVHN"
train_set_name = "FashionMNIST"
# train_set_name = "MNIST"
# train_set_name = "CELEBA"
train_set_name = "CIFAR"


# ==============PREPARE DATA==========================
shuffle = True

data_transform = transforms.Compose([transforms.ToTensor()])
train_batch_size = 100
test_samples = 100

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
        shuffle=True,
    )

    ood_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "data-svhn", split="test", download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=True,
    )
elif train_set_name == "FashionMNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "data-fashion-mnist", train=True, download=True, transform=data_transform
        ),
        batch_size=train_batch_size,
        shuffle=shuffle,
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
            "data-mnist", train=False, download=True, transform=data_transform
        ),
        batch_size=test_samples,
        shuffle=False,
    )


class PatchEmbed(torch.nn.Module):
    """Split image into patches and then embed them.
    Parameters
    ----------
    img_size : int
        Size of the image (it is a square).
    patch_size : int
        Size of the patch (it is a square).
    in_chans : int
        Number of input channels.
    embed_dim : int
        The emmbedding dimension.
    Attributes
    ----------
    n_patches : int
        Number of patches inside of our image.
    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding.
    """

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = torch.nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x


class PatchEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
    ):
        self.patch_size = patch_size
        super().__init__()
        self.projection = torch.nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            torch.nn.Conv2d(
                in_channels, emb_size, kernel_size=patch_size, stride=patch_size
            ),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = torch.nn.Parameter(
            torch.randn((img_size // patch_size) ** 2 + 1, emb_size)
        )

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


# PatchEmbedding()(x).shape
# patch_embedding_layer = PatchEmbedding(32, 4)

patch_embedding_layer = PatchEmbedding(img_size=32, patch_size=16, emb_size=48)
train_data = next(iter(train_loader))[0]
id_data = next(iter(test_loader))[0]
ood_data = next(iter(ood_loader))[0]


patch_emb = patch_embedding_layer(id_data)

print(id_data.shape)
print(patch_emb.shape)

joe = torch.nn.Conv2d(in_channels=3, out_channels=200, kernel_size=4, stride=4)
print(joe(id_data).shape)


encoder_layer = torch.nn.TransformerEncoderLayer(
    d_model=48, nhead=8, batch_first=True, dim_feedforward=2048
)
out = encoder_layer(patch_emb)
out = encoder_layer(out)

print(out.shape)


class TransformerAutoencoder(torch.nn.Module):
    def __init__(
        self,
        d_model=128,
        nhead=8,
        dim_feedforward=2048,
        n_modules=1,
        input_shape=(3, 32, 32),
    ):
        super(TransformerAutoencoder, self).__init__()
        self.patch_embed = PatchEmbedding(img_size=32, patch_size=4, emb_size=d_model)

        self.seq = torch.nn.Sequential(
            *[
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    batch_first=True,
                    dim_feedforward=dim_feedforward,
                )
                for i in range(n_modules)
            ],
            Flatten(),
            torch.nn.Linear(8320, np.product(input_shape)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.seq(x)
        x = x.view(-1, 3, 32, 32)
        return x

    def pred_loss(self, x, return_mean=True):
        y_pred = self.forward(x)
        if return_mean:
            return ((x - y_pred) ** 2).mean(-1).mean(-1).mean(-1)
        else:
            return (x - y_pred) ** 2


lr = 0.001
trans_ae = TransformerAutoencoder()
optim = torch.optim.Adam(trans_ae.parameters(), lr=lr)
num_epochs = 1
criterion = torch.nn.MSELoss()

for i in range(num_epochs):
    for train_data, _ in train_loader:
        outp = trans_ae(train_data)
        loss = criterion(train_data, outp)

        optim.zero_grad()
        loss.backward()
        optim.step()

    print(loss.item())

id_nll = trans_ae.pred_loss(id_data).detach().cpu().numpy()
ood_nll = trans_ae.pred_loss(ood_data).detach().cpu().numpy()

id_pred = trans_ae(id_data).detach().cpu().numpy()
ood_pred = trans_ae(ood_data).detach().cpu().numpy()
print(calc_auroc(id_nll, ood_nll))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 10)
for i, ax in enumerate(axes):
    ax.imshow(np.moveaxis(id_data[i].detach().cpu().numpy(), 0, -1))

fig, axes = plt.subplots(1, 10)
for i, ax in enumerate(axes):
    ax.imshow(np.moveaxis(id_pred[i], 0, -1))

fig, axes = plt.subplots(1, 10)
for i, ax in enumerate(axes):
    ax.imshow(np.moveaxis(ood_data[i].detach().cpu().numpy(), 0, -1))
fig, axes = plt.subplots(1, 10)
for i, ax in enumerate(axes):
    ax.imshow(np.moveaxis(ood_pred[i], 0, -1))
