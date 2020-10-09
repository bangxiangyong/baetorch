import torch
import torch.nn as nn
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.cluster import normalized_mutual_info_score

from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, DenseLayers, infer_decoder, Autoencoder
from baetorch.baetorch.models.cholesky_layer import CholLayer
from baetorch.baetorch.plotting import plot_contour, get_grid2d_latent
from baetorch.baetorch.util.seed import bae_set_seed

bae_set_seed(321)

#=========load data
from sklearn.datasets import load_iris, make_blobs, make_moons

# iris_data = load_iris()
# x_train = iris_data['data']
# y_train = iris_data['target']
# x_train, y_train = make_moons(n_samples=500)
x_train, y_train = make_blobs(n_samples=100, centers=5, n_features=2)

x_train = MinMaxScaler().fit_transform(x_train)

x_train_torch = torch.tensor(x_train).float()
y_train_torch = torch.tensor(y_train).int()


#=======AutoEncoder
latent_dim = 100
pretrain_epoch =150
num_epoch = 250
num_clusters= 3
encoder = Encoder([DenseLayers(architecture=[100],
                               output_size=latent_dim,
                               input_size=x_train.shape[-1],
                               activation='leakyrelu',
                               last_activation='none')])
decoder_mu = infer_decoder(encoder, last_activation='none')
decoder_sig = CholLayer(architecture=[100], input_size=latent_dim, output_size=x_train.shape[-1])
autoencoder = Autoencoder(encoder=encoder, decoder_mu=decoder_mu, decoder_sig=decoder_sig)

#=========cluster layer=====
gamma = 10000
batch_size = x_train.shape[0]
num_samples = 5
learning_rate = 0.001

#======training====
bae_ensemble_full_cov = BAE_Ensemble(autoencoder=autoencoder, cluster_weight=gamma,
                                     num_samples=num_samples, learning_rate=learning_rate,
                                     likelihood="full_gaussian")

#pretrain to optimise reconstruction loss
bae_ensemble_full_cov.fit(x_train, num_epochs=pretrain_epoch, mode="mu")
bae_ensemble_full_cov.fit(x_train, num_epochs=num_epoch, mode="sigma")


grid, grid_2d=get_grid2d_latent(x_train)
nll_full = bae_ensemble_full_cov.predict_samples(grid_2d, select_keys=["nll_sigma"])
y_sig = bae_ensemble_full_cov.predict_samples(grid_2d, select_keys=["y_sigma"])
se = bae_ensemble_full_cov.predict_samples(grid_2d, select_keys=["se"])
nll_full = np.clip(nll_full,-100,100)

fig, ((ax1,ax2),(ax3,ax4), (ax5,ax6)) = plt.subplots(3,2)
plot_contour(-nll_full.mean(0)[0],grid, fig=fig, ax=ax1, colorbar=False)
plot_contour(nll_full.std(0)[0],grid, fig=fig, ax=ax2, colorbar=False)
plot_contour(y_sig.mean(0)[0].mean(-1),grid, fig=fig, ax=ax3, colorbar=False)
plot_contour(y_sig.std(0)[0].mean(-1),grid, fig=fig, ax=ax4, colorbar=False)
plot_contour(se.mean(0)[0].mean(-1),grid, fig=fig, ax=ax5, colorbar=False)
plot_contour(se.std(0)[0].mean(-1),grid, fig=fig, ax=ax6, colorbar=False)
ax1.scatter(x_train[:,0],x_train[:,1], c=y_train)
ax2.scatter(x_train[:,0],x_train[:,1], c=y_train)
ax3.scatter(x_train[:,0],x_train[:,1], c=y_train)
ax4.scatter(x_train[:,0],x_train[:,1], c=y_train)
ax5.scatter(x_train[:,0],x_train[:,1], c=y_train)
ax6.scatter(x_train[:,0],x_train[:,1], c=y_train)

for ax_i,title_i in zip([ax1,ax2,ax3,ax4,ax5,ax6],
                        ["NLL MEAN", "NLL STD", "Y SIG MEAN", "Y SIG STD", "SE MEAN","SE STD"]):
    ax_i.set_title(title_i)

