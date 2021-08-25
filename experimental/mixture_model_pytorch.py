import torch.distributions as D
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

from baetorch.baetorch.util.seed import bae_set_seed

n_components = 2
mix = D.Categorical(
    torch.ones(
        n_components,
    )
)
comp = D.Normal(
    torch.randn(
        n_components,
    ),
    torch.rand(
        n_components,
    ),
)
gmm = D.MixtureSameFamily(mix, comp)

bae_set_seed(123)


class Autoencoder(torch.nn.Module):
    def __init__(self):
        self.layer1 = torch.nn.Linear(2, 100)
        self.layer2 = torch.nn.Linear(100, 100)
        self.layer3 = torch.nn.Linear(100, 2)

        self.gmm_layer = torch.nn.Linear(1, 2)

    def forward(self, x):
        y_recon = self.layer3(self.layer2(self.layer1(x)))
        recon_loss = ((x - y_recon) ** 2).mean()
        return x


class GMM_Layer(torch.nn.Module):
    def __init__(self, input_dim=1, k_components=2, hidden_dim=2):
        super(GMM_Layer, self).__init__()
        # self.weight_layer = torch.nn.Linear(input_dim, k_components)
        self.weight_layer = torch.nn.Sequential(
            *[
                torch.nn.Linear(input_dim, hidden_dim, bias=True),
                torch.nn.SELU(),
                torch.nn.Linear(hidden_dim, k_components, bias=True),
            ]
        )

        self.means_layer = [
            torch.nn.Sequential(
                *[
                    torch.nn.Linear(input_dim, hidden_dim, bias=False),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(hidden_dim, input_dim, bias=False),
                ]
            )
            for i in range(k_components)
        ]

        self.stdevs_layer = [
            torch.nn.Sequential(
                *[
                    torch.nn.Linear(input_dim, hidden_dim, bias=False),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(hidden_dim, input_dim, bias=False),
                    torch.nn.Softplus(),
                ]
            )
            for i in range(k_components)
        ]
        self.gll_loss = torch.nn.GaussianNLLLoss()
        # self.means = torch.tensor(
        #     np.random.randn(k_components, input_dim), requires_grad=True
        # )
        # self.stdevs = torch.tensor(
        #     np.abs(np.random.randn(k_components, input_dim)), requires_grad=True
        # )

    def forward(self, x):
        weights = self.forward_weights(x)
        means, stdevs = self.forward_means_stds(x)

        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(means, stdevs), 1)
        gmm = D.MixtureSameFamily(mix, comp)

        return -gmm.log_prob(x).mean()

    def forward_weights(self, x):
        return F.softmax(self.weight_layer(x), dim=1)

    def forward_means_stds(self, x):
        means = torch.moveaxis(
            torch.stack([mean_layer(x) for mean_layer in self.means_layer]),
            (0, 1, 2),
            (1, 0, 2),
        )
        stddevs = torch.moveaxis(
            torch.stack([std_layer(x) for std_layer in self.stdevs_layer]),
            (0, 1, 2),
            (1, 0, 2),
        )
        return means, stddevs


import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim
import torch.distributions as D

scaler = MinMaxScaler()
gmm = GMM_Layer(k_components=2)

optimiser = optim.Adam(gmm.parameters(), lr=0.000001)


x1 = torch.distributions.Normal(loc=-0.25, scale=0.15).sample([125])
x2 = torch.distributions.Normal(loc=-0.25, scale=0.15).sample([100])
x = torch.from_numpy(
    scaler.fit_transform(np.expand_dims(np.concatenate((x1, x2)), 1))
).float()

x_eval = np.linspace(-1, 1, 100)
x_eval_torch = torch.from_numpy(scaler.transform(np.expand_dims(x_eval, 1))).float()

num_iter = 1
for i in range(num_iter):
    # mix = D.Categorical(weights)
    # comp = D.Independent(D.Normal(means, stdevs), 1)
    # gmm = D.MixtureSameFamily(mix, comp)

    optimiser.zero_grad()
    loss = gmm(x)
    loss.backward()
    optimiser.step()

    print(loss.item())

x1_np = x1.detach().numpy()
x2_np = x2.detach().numpy()
prob_pred = gmm.forward_weights(x_eval_torch).detach().numpy()

fig, ax1 = plt.subplots(1, 1)
ax1.hist(x1_np, alpha=0.5, density=True)
ax1.hist(x2_np, alpha=0.5, density=True)
ax1.hist(np.concatenate((x1_np, x2_np)), alpha=0.5, density=True)
ax2 = ax1.twinx()

# for k in range(prob_pred.shape[1]):
#     ax2.plot(x_eval, prob_pred[:, k])

# print(gmm.means)
# print(gmm.stdevs)

gm = GaussianMixture(n_components=2, random_state=0).fit(
    scaler.fit_transform(np.expand_dims(np.concatenate((x1, x2)), 1))
)
x_np_scale = scaler.transform(np.expand_dims(x_eval, 1))

proba_x = gm.predict_proba(x_np_scale)

for k in range(proba_x.shape[1]):
    ax2.plot(x_eval, proba_x[:, k])
