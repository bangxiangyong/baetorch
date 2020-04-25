import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.stats as stats
from scipy.signal import find_peaks
from math import log10, floor


seed_value=125
torch.manual_seed(seed_value)
np.random.seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True  #tested - needed for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed_value)
from bnn.develop.bayesian_autoencoders.base_autoencoder import *
from torchvision import datasets, transforms
from bnn.develop.bayesian_autoencoders.bae_mcdropout import BAE_MCDropout
from bnn.develop.bayesian_autoencoders.bae_vi import BAE_VI, VAE
from bnn.develop.bayesian_autoencoders.bae_ensemble import BAE_Ensemble
from bnn.develop.bayesian_autoencoders.plotting import *
from bnn.develop.bayesian_autoencoders.evaluation import *
from bnn.develop.bayesian_autoencoders.test_suite import run_test_model
from bnn.develop.bayesian_autoencoders.lr_range_finder_class import run_auto_lr_range
import time

#EXAMPLE MAIN
#load fashion mnist
test_samples= 1000
train_batch_size = 100
data_transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data-fashion-mnist', train=True, download=True, transform=data_transform), batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data-fashion-mnist', train=False, download=True, transform=data_transform), batch_size=test_samples, shuffle=True)
ood_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data-mnist', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor(),])), batch_size=test_samples, shuffle=True)

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data-mnist', train=True, download=True, transform=data_transform), batch_size=train_batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data-mnist', train=False, download=True, transform=data_transform), batch_size=test_samples, shuffle=True)
# ood_loader = torch.utils.data.DataLoader(
#     datasets.FashionMNIST('data-fashion-mnist', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor(),])), batch_size=test_samples, shuffle=True)
#

#get some samples
x_train, x_train_label = get_sample_dataloader(train_loader)
x_test, x_test_label = get_sample_dataloader(test_loader)
x_ood, x_ood_label = get_sample_dataloader(ood_loader)

#model architecture
latent_dim = 10
input_dim = 28
input_channel = 1

# conv_architecture=[input_channel,32,64,128]
# conv_architecture=[input_channel,10,32,64]
conv_architecture=[input_channel,12,24,48]

#specify encoder
#with convolutional layers
encoder = Encoder([ConvLayers(input_dim=input_dim,conv_architecture=conv_architecture, conv_kernel=[4,4,4], conv_stride=[2,2,2], last_activation="sigmoid"),
           DenseLayers(architecture=[],output_size=latent_dim)])

#only dense layers
# encoder = Encoder([DenseLayers(input_size=flatten_torch(x_train).shape[-1],architecture=[500,300],output_size=latent_dim)])

#specify decoder-mu
decoder_mu = infer_decoder(encoder,last_activation="sigmoid") #symmetrical to encoder

#specify decoder-sigma
decoder_sig_inferred = infer_decoder(encoder, last_activation="none") #inferred as a reflection of encoder
decoder_sig_dense = DenseLayers(input_size=latent_dim,
                                output_size=input_channel*input_dim*input_dim,
                                architecture=[800]) #dense layers only

# TBA: Clustering layer
# decoder_cluster = DenseLayers(architecture=[100,5], input_size=1, output_size=1)

#combine them into autoencoder
autoencoder_1 = Autoencoder(encoder, decoder_mu, decoder_sig_inferred) #option 1
autoencoder_2 = Autoencoder(encoder, decoder_mu, decoder_sig_dense) #option 2
autoencoder_3 = Autoencoder(encoder, decoder_mu) #option 3

#forward pass
encoded = encoder(x_train)
decoded_mu = decoder_mu(encoded)

print(encoded)
print(decoded_mu)
print(autoencoder_1(x_train)[1].shape)
print(autoencoder_2(x_train)[1].shape)

#BAE STARTS HERE
#CONVERT TO BAE
homoscedestic_mode = "none"

# bae_mcdropout = BAE_MCDropout(autoencoder=autoencoder_2, dropout_p=0.2,
#                               num_train_samples=1,num_samples=100, use_cuda=True,
#                               weight_decay=1, learning_rate=0.002,
#                               num_epochs=10, homoscedestic_mode=homoscedestic_mode)

# bae_ensemble = BAE_Ensemble(autoencoder=autoencoder_1, use_cuda=True, anchored=True,
#                             weight_decay=1, learning_rate=0.0001, learning_rate_sig=0.0001,
#                             num_samples=5, homoscedestic_mode=homoscedestic_mode)
#

# bae_vi = BAE_VI(autoencoder=autoencoder_3,
#                               num_train_samples=5,num_samples=25, use_cuda=True, anchored=False,
#                               weight_decay=1, learning_rate=0.0005,
#                               num_epochs=10, homoscedestic_mode=homoscedestic_mode)

vae = VAE(autoencoder=autoencoder_3,
                              num_train_samples=5,num_samples=50, use_cuda=True, anchored=False,
                              weight_decay=0.01, learning_rate=0.005,
                              num_epochs=10, homoscedestic_mode=homoscedestic_mode)

bae_model = vae

# bae_vi.fit(x_train, num_epochs=1, mode="mu")
# vae.fit(x_train, num_epochs=1, mode="mu")

# bae_vi.learning_rate = 0.00025
# bae_vi.fit(train_loader, num_epochs=20, mode="mu")
# vae.fit(x_train, num_epochs=4000, mode="mu")

# encoded = bae_vi.autoencoder.encoder(x_train.cuda())
# decoded_mu = bae_vi.autoencoder.decoder_mu(encoded)[0]
# decoded_sig = bae_vi.autoencoder.decoder_sig(encoded)[0]
# bae_ensemble.learning_rate = 0.00025
# bae_ensemble.learning_rate = 0.005
# bae_ensemble.fit(train_loader, num_epochs=20, mode="mu")
# bae_ensemble.fit(train_loader, num_epochs=5, mode="sigma", sigma_train="separate")
# bae_ensemble.fit(train_loader, num_epochs=5, mode="sigma", sigma_train="joint")


# bae_ensemble.learning_rate = 0.001
# bae_ensemble.fit_one(x_train, num_epochs=1)


#train mu network
run_auto_lr_range(train_loader, bae_model, mode="mu", sigma_train="separate")
# bae_model.fit(train_loader,num_epochs=5, mode="mu")
#
# if bae_model.decoder_sigma_enabled:
#     run_auto_lr_range(train_loader, bae_model, mode="sigma",sigma_train="separate", reset_params=False)
#     bae_model.fit(train_loader,num_epochs=1, mode="sigma", sigma_train="separate")
#
# #for each model, evaluate and plot:
# bae_models = [bae_model]
# id_data_test = test_loader
# ood_data_list = [ood_loader]
# ood_data_names = ["OOD"]
# exclude_keys =[]
# dist_exclude_keys = ["aleatoric_var","waic_se","nll_homo_var","waic_homo","waic_sigma"]
# train_set_name = "FashionMNIST"
#
# #run evaluation test of model on ood data set
# run_test_model(bae_models=bae_models, id_data_test=test_loader, ood_data_list=ood_data_list,
#                ood_data_names=ood_data_names, id_data_name=train_set_name)
