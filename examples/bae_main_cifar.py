import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.stats as stats

seed_value=100
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

train_set_name = "CIFAR"

#EXAMPLE MAIN
#load CIFAR
test_samples= 1000
train_batch_size = 25
data_transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data-cifar', train=True, download=True,
                   transform=data_transform
                   ), batch_size=train_batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data-cifar', train=False, download=True,
                   transform=data_transform
                   ), batch_size=test_samples, shuffle=True)

ood_loader = torch.utils.data.DataLoader(
    datasets.SVHN('data-svhn', split="test", download=True,
                   transform=data_transform
                   ), batch_size=test_samples, shuffle=True)

# train_loader = torch.utils.data.DataLoader(
#     datasets.SVHN('data-svhn', split="train", download=True,
#                    transform=data_transform
#                    ), batch_size=train_batch_size, shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(
#     datasets.SVHN('data-svhn', split="test", download=True,
#                    transform=data_transform
#                    ), batch_size=test_samples, shuffle=True)
#
# ood_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('data-cifar', train=False, download=True,
#                    transform=data_transform
#                    ), batch_size=test_samples, shuffle=True)



#get some samples
x_train, x_train_label = get_sample_dataloader(train_loader)
x_test, x_test_label = get_sample_dataloader(test_loader)
x_ood, x_ood_label = get_sample_dataloader(ood_loader)

#model architecture
latent_dim = 1024
input_dim = 32
input_channel = 3

# conv_architecture=[input_channel,32,64,128]
conv_architecture=[input_channel,12,24,48]

#specify encoder
encoder = Encoder([ConvLayers(input_dim=input_dim,conv_architecture=conv_architecture, conv_kernel=[4,4,4], conv_stride=[2,2,2], last_activation="sigmoid"),
           DenseLayers(architecture=[1024],output_size=latent_dim)])

#specify decoder-mu
decoder_mu = infer_decoder(encoder,last_activation="sigmoid") #symmetrical to encoder

#specify decoder-sigma
decoder_sig_conv = infer_decoder(encoder,last_activation="none") #with conv layers
decoder_sig_dense = DenseLayers(input_size=latent_dim,
                                output_size=input_channel*input_dim*input_dim,
                                architecture=[800]) #dense layers only

# TBA: Clustering layer
# decoder_cluster = DenseLayers(architecture=[100,5], input_size=1, output_size=1)

#combine them into autoencoder
autoencoder_1 = Autoencoder(encoder, decoder_mu, decoder_sig_conv) #option 1
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
# bae_mcdropout = BAE_MCDropout(autoencoder=autoencoder_1, dropout_p=0.2,
#                               num_train_samples=10,num_samples=100, use_cuda=True,
#                               weight_decay=1, learning_rate=0.002,
#                               num_epochs=10, homoscedestic_mode=homoscedestic_mode)

bae_ensemble = BAE_Ensemble(autoencoder=autoencoder_1, use_cuda=True, anchored=False,
                            weight_decay=0, learning_rate=0.001, learning_rate_sig=0.001,
                            num_samples=1, homoscedestic_mode=homoscedestic_mode)

# bae_vi = BAE_VI(autoencoder=autoencoder_1,
#                               num_train_samples=1,num_samples=25, use_cuda=True, anchored=False,
#                               weight_decay=1, learning_rate=0.0005,
#                               num_epochs=10, homoscedestic_mode=homoscedestic_mode)

# vae = VAE(autoencoder=autoencoder_1,
#                               num_train_samples=1,num_samples=1, use_cuda=True, anchored=False,
#                               weight_decay=1, learning_rate=0.01, homoscedestic_mode=homoscedestic_mode)

# bae_vi.fit(x_train, num_epochs=1, mode="mu")
# vae.fit(x_train, num_epochs=1, mode="mu")

# bae_vi.fit(x_train, num_epochs=4000, mode="mu")
# vae.fit(x_train, num_epochs=4000, mode="mu")
# bae_ensemble.fit(x_train, num_epochs=1, mode="mu")

# encoded = bae_vi.autoencoder.encoder(x_train.cuda())
# decoded_mu = bae_vi.autoencoder.decoder_mu(encoded)[0]
# decoded_sig = bae_vi.autoencoder.decoder_sig(encoded)[0]

bae_ensemble.learning_rate = 0.0005
bae_ensemble.fit(train_loader, num_epochs=20, mode="mu")
# bae_ensemble.fit(train_loader, num_epochs=5, mode="sigma", sigma_train="separate")
# bae_ensemble.fit(train_loader, num_epochs=5, mode="sigma", sigma_train="joint")


# bae_ensemble.fit(x_train, num_epochs=1000, mode="mu")
# bae_ensemble.fit(x_train, num_epochs=50, mode="sigma", sigma_train="separate")
# bae_mcdropout.fit(x_train, num_epochs=1000, mode="mu")
# bae_mcdropout.fit(x_train, num_epochs=100, mode="sigma", sigma_train="separate")
# bae_mcdropout.use_cuda
# bae_ensemble.fit(x_train, num_epochs=50, mode="sigma", sigma_train="separate")
# bae_mcdropout.fit(x_train, num_epochs=100, mode="mu")
# bae_mcdropout.fit(x_train, num_epochs=50, mode="sigma", sigma_train="separate")

# predict_test = bae_ensemble.predict(x_test)
# predict_ood = bae_ensemble.predict(x_ood)

# sample_outputs = bae_mcdropout.predict(x_train)

# print(bae_mcdropout.autoencoder.log_noise.detach().cpu().numpy())
# print(bae_mcdropout.get_homoscedestic_noise()[0].mean())

#
#


#for each model, evaluate and plot:
# bae_models = [bae_vi,vae]
# bae_models = [vae]
bae_models = [bae_ensemble]
# bae_models = [bae_vi,vae]
# bae_models = [bae_mcdropout]
# bae_model_names = ["ENSEMBLE"]
# bae_models = [bae_ensemble,bae_mcdropout]
# bae_model_names = ["MC","ENSEMBLE"]
# id_data_test = x_test
id_data_test = test_loader
# ood_data_list = [x_ood]
ood_data_list = [ood_loader]
ood_data_names = ["OOD"]
exclude_keys =[]
dist_exclude_keys = ["aleatoric_var","waic_se","nll_homo_mean","nll_homo_var","waic_homo","waic_sigma"]

# predict_test = bae_mcdropout.predict(id_data_test)

for bae_model in bae_models:
    #compute model outputs
    predict_test = bae_model.predict(id_data_test)
    predict_ood_list = [bae_model.predict(ood_data) for ood_data in ood_data_list]

    #plot reconstruction image of test set
    plot_samples_img(data=predict_test, reshape_size=(32,32,3), savefile=bae_model.model_name +"-"+"TEST"+"-samples"+".png")

    #evaluate performance curves by comparing against OOD datasets
    for predict_ood,ood_data_name in zip(predict_ood_list,ood_data_names):

        plot_roc_curve(predict_test,predict_ood, title=bae_model.model_name +"-"+ood_data_name, savefile=bae_model.model_name +"-"+ood_data_name+"-AUROC"+".png")
        plot_prc_curve(predict_test,predict_ood, title=bae_model.model_name +"-"+ood_data_name, savefile=bae_model.model_name +"-"+ood_data_name+"-AUPRC"+".png")

        #evaluation
        auroc_list, fpr80_list, metric_names= calc_auroc(predict_test,predict_ood, exclude_keys=exclude_keys)
        auprc_list, metric_names= calc_auprc(predict_test,predict_ood, exclude_keys=exclude_keys)
        plot_samples_img(data=predict_ood, reshape_size=(32,32,3), savefile=bae_model.model_name +"-"+ood_data_name+"-samples"+".png")
        #save performance results as csv
        save_csv_metrics(train_set_name+"_"+ood_data_name,bae_model.model_name , auroc_list, auprc_list, fpr80_list, metric_names)

    #plot and compare distributions of data per model
    plot_output_distribution(predict_test,*predict_ood_list,legends=["TEST"]+ood_data_names,exclude_keys=dist_exclude_keys)
    output_means, output_medians, dist_metric_names = calc_variance_dataset(predict_test,*predict_ood_list,legends=["TEST"]+ood_data_names, exclude_keys=["aleatoric_var","waic_se","nll_homo_mean","nll_homo_var","waic_homo","waic_sigma"])
    save_csv_distribution(train_set_name,bae_model.model_name, output_means, output_medians, dist_metric_names)

    homoscedestic_noise = bae_model.get_homoscedestic_noise()
    print(bae_model.model_name +" Homo-noise:"+str(homoscedestic_noise))

    plot_calibration_curve(bae_model, id_data_test)
    # plot_latent(x_test, x_ood, legends=["TEST"]+ood_data_names, model=bae_model, transform_pca=True)
    plot_latent(x_test, x_ood, legends=["TEST"]+ood_data_names, model=bae_model, transform_pca=False)
