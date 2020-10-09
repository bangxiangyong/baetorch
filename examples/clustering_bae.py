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
from baetorch.baetorch.util.seed import bae_set_seed

bae_set_seed(321)

# Clustering layer definition (see DCEC article for equations)
class ClusteringLayer(nn.Module):
    def __init__(self, architecture=[], activation='tanh', last_activation='none', input_size=10, output_size=10, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha

        if len(architecture) != 0:
            if len(architecture) == 1:
                self.dense_decoder = DenseLayers(architecture=[],
                                                           output_size=architecture[-1],
                                                           input_size=self.input_size,
                                                           activation=activation,
                                                           last_activation=last_activation)
            else:
                self.dense_decoder = DenseLayers(architecture=architecture[:-1],
                                                           output_size=architecture[-1],
                                                           input_size=self.input_size,
                                                           activation=activation,
                                                           last_activation=last_activation)
            self.weight = nn.Parameter(torch.Tensor(self.output_size, architecture[-1]))
        else:
            self.dense_decoder = None
            self.weight = nn.Parameter(torch.Tensor(self.output_size, self.input_size))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = self.dense_layers(x)
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)

def target(out_distr):
    tar_dist = out_distr ** 2 / torch.sum(out_distr, axis=0)
    tar_dist = torch.transpose(torch.transpose(tar_dist,0,1) / torch.sum(tar_dist, axis=1),0,1)

    return tar_dist

#=========load data
from sklearn.datasets import load_iris, make_blobs

# iris_data = load_iris()
# x_train = iris_data['data']
# y_train = iris_data['target']
# x_train, y_train = make_blobs(n_samples=100, centers=3, n_features=2, cluster_std=(2.,2.,0.5))
x_train, y_train = make_blobs(n_samples=500, centers=3, n_features=2)

x_train = MinMaxScaler().fit_transform(x_train)

x_train_torch = torch.tensor(x_train).float()
y_train_torch = torch.tensor(y_train).int()

#=======AutoEncoder
latent_dim = 2
pretrain_epoch =150
num_epoch = 1000
num_clusters= 3
encoder = Encoder([DenseLayers(architecture=[300],
                               output_size=latent_dim,
                               input_size=x_train.shape[-1],
                               activation='leakyrelu',
                               last_activation='none')])
decoder = infer_decoder(encoder, last_activation='none')
dense_cluster = DenseLayers(architecture=[300],
                               output_size=latent_dim,
                               input_size=latent_dim,
                               activation='tanh',
                               last_activation='none')
decoder_cluster = ClusteringLayer(architecture=[123], input_size=latent_dim, output_size=num_clusters)
autoencoder = Autoencoder(encoder=encoder, decoder_mu=decoder, decoder_cluster=decoder_cluster)

#=========cluster layer=====
gamma = 10000
batch_size = x_train.shape[0]
num_samples = 1
learning_rate = 0.001
#======training====
mse_loss_func = nn.MSELoss()
kl_div_func = nn.KLDivLoss()

latent_output_samples = []
cluster_predictions_samples = []

bae_ensemble_clustering = BAE_Ensemble(autoencoder=autoencoder, cluster_weight=gamma, num_samples=num_samples, learning_rate=learning_rate)
# run_auto_lr_range_v2(x_train, bae_ensemble_clustering)
bae_ensemble_clustering.cluster_weight = 0
bae_ensemble_clustering.fit(x_train, num_epochs=pretrain_epoch)
bae_ensemble_clustering.cluster_weight = gamma
bae_ensemble_clustering.fit(x_train, num_epochs=num_epoch)

latent_output_samples = np.array(latent_output_samples)
cluster_predictions_samples = np.array(cluster_predictions_samples)

y_cluster = bae_ensemble_clustering.predict_cluster(x_train)


#plot latent dimension
#
# fig,axes1 = plt.subplots(2,5, figsize=(10,6))
# fig,axes2 = plt.subplots(2,5, figsize=(10,6))
# axes1 = axes1.flatten()
# axes2 = axes2.flatten()
#
# for i in range(num_plots):
#     axes1[i].scatter(latent_output_samples[i][:,0],latent_output_samples[i][:,1], c=y_train)
#     axes1[i].set_title("TRUE EPOCH:{}".format(plot_intervals[i]))
#     axes2[i].scatter(latent_output_samples[i][:,0],latent_output_samples[i][:,1], c=cluster_predictions_samples[i])
#     axes2[i].set_title("PRED EPOCH:{}".format(plot_intervals[i]))
#
# from sklearn.metrics.cluster import normalized_mutual_info_score
# nmi_res = normalized_mutual_info_score(y_train,cluster_predictions_samples[-1])
# print(nmi_res)

#======plot contour map====
def plot_latent_contour(x_input, model, xtype='latent',return_type = 'epistemic'):
    #if x_input is in latent space, don't have to encode it again
    if xtype == 'latent':
        latent_space = model.predict_latent(x_input,transform_pca=False)[0]
        encode = False
    elif xtype == 'input':
        latent_space = x_input
        encode = True
    span =0.5
    # latent_space = latent_output_samples[-1]
    grid = np.mgrid[latent_space[:,0].min()-span:latent_space[:,0].max()+span:100j,
           latent_space[:,1].min()-span:latent_space[:,1].max()+span:100j]
    grid_2d = grid.reshape(2, -1).T

    #compute on contour
    # prob_contour = model(torch.from_numpy(grid_2d).float()).detach().numpy()
    # if isinstance(model, ClusteringLayer):
    #     prob_contour = np.log(prob_contour.max(1))
    # else:
    #     prob_contour = ((prob_contour-grid_2d)**2).mean(-1)
    # p_hat = model.predict_cluster(grid_2d, encode=True)
    y_pred = model.predict_cluster(grid_2d, encode=encode)
    # p_hat = np.max(y_pred,axis=-1)
    epistemic = (np.mean(y_pred**2, axis=0) - np.mean(y_pred, axis=0)**2).max(-1)
    aleatoric = np.mean(y_pred*(1-y_pred), axis=0).max(-1)
    mean_prob = np.mean(y_pred,axis=0).max(-1)
    return_data = {"epistemic":epistemic,"aleatoric":aleatoric,"mean":mean_prob}
    cmaps = {"epistemic":'Greys',"aleatoric":'Greys',"mean":'Greys_r'}
    # prob_contour = mean_prob
    # prob_contour = epistemic
    # prob_contour = aleatoric
    prob_contour = return_data[return_type]
    levels = np.linspace(prob_contour.min()*10,prob_contour.max()*10,25)

    fig, ax = plt.subplots(figsize=(16, 9))
    # cmap = sns.cubehelix_palette(light=0, as_cmap=True)
    contour = ax.contourf(grid[0], grid[1], prob_contour.reshape(100, 100)*10, levels=levels, cmap=cmaps[return_type])
    fig.colorbar(contour)
    scatter_lifetime=ax.scatter(latent_space[:,0],latent_space[:,1], c=y_train, cmap='viridis')
    return prob_contour

def calculate_nmi(x,y,model):
    # for unsupervised classification
    y_preds = np.argmax(model.predict_cluster(x),-1)
    nmi_res = np.array([normalized_mutual_info_score(y,y_pred) for y_pred in y_preds]).mean(0)
    return nmi_res

# latent_space = bae_ensemble_clustering.predict_latent(x_train,transform_pca=False)
# return_type = "a"
# prob_cont=plot_latent_contour(x_train, bae_ensemble_clustering, xtype='latent')
nmi = calculate_nmi(x_train,y_train,bae_ensemble_clustering)
print(nmi)
# xtype = 'input'
xtype = 'input'
prob_cont=plot_latent_contour(x_train, bae_ensemble_clustering,  xtype=xtype, return_type='mean')
prob_cont=plot_latent_contour(x_train, bae_ensemble_clustering,  xtype=xtype, return_type='aleatoric')
prob_cont=plot_latent_contour(x_train, bae_ensemble_clustering,  xtype=xtype, return_type='epistemic')

# plot_latent_contour(latent_output_samples[-1], decoder)

# bae_ensemble_clustering.predict_latent(x_train)




