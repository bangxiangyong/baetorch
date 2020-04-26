import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.stats as stats
from baetorch.models.base_autoencoder import *
from torchvision import datasets, transforms
from baetorch.models.bae_mcdropout import BAE_MCDropout
from baetorch.models.bae_vi import BAE_VI, VAE
from baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.plotting import *
from baetorch.evaluation import *
from baetorch.util.seed import bae_set_seed

bae_set_seed(100)

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
latent_dim = 1024
input_dim = 28
input_channel = 1

# conv_architecture=[input_channel,32,64,128]
# conv_architecture=[input_channel,10,32,64]
conv_architecture=[input_channel,12,24,48]
#specify encoder
encoder = Encoder([ConvLayers(input_dim=input_dim,conv_architecture=conv_architecture, conv_kernel=[4,4,4], conv_stride=[2,2,2], last_activation="sigmoid"),
           DenseLayers(architecture=[],output_size=latent_dim)])

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

bae_ensemble = BAE_Ensemble(autoencoder=autoencoder_3, use_cuda=True, anchored=False,
                            weight_decay=1, learning_rate=0., learning_rate_sig=0.001,
                            num_samples=5, homoscedestic_mode=homoscedestic_mode)


# bae_vi = BAE_VI(autoencoder=autoencoder_3,
#                               num_train_samples=1,num_samples=25, use_cuda=True, anchored=False,
#                               weight_decay=1, learning_rate=0.0005,
#                               num_epochs=10, homoscedestic_mode=homoscedestic_mode)

# vae = VAE(autoencoder=autoencoder_1,
#                               num_train_samples=1,num_samples=50, use_cuda=True, anchored=False,
#                               weight_decay=1, learning_rate=0.005,
#                               num_epochs=10, homoscedestic_mode=homoscedestic_mode)

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
bae_ensemble.fit(train_loader, num_epochs=10, mode="mu")
# bae_ensemble.fit(train_loader, num_epochs=5, mode="sigma", sigma_train="separate")
# bae_ensemble.fit(train_loader, num_epochs=5, mode="sigma", sigma_train="joint")

# bae_ensemble.learning_rate = 0.001
# bae_ensemble.fit(x_train, num_epochs=4000, mode="mu")
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
# bae_models = [bae_vi]
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
dist_exclude_keys = ["aleatoric_var","waic_se","nll_homo_var","waic_homo","waic_sigma"]
train_set_name = "FashionMNIST"
# train_set_name = "MNIST"

# predict_test = bae_mcdropout.predict(id_data_test)

def remove_nan(predict_res):
    predict_res['bce_mean'] = np.nan_to_num(predict_res['bce_mean'])
    predict_res['bce_var'] = np.nan_to_num(predict_res['bce_var'])
    predict_res['bce_waic'] = np.nan_to_num(predict_res['bce_waic'])
    return predict_res

for bae_model in bae_models:

    #compute model outputs
    predict_test = bae_model.predict(id_data_test)
    # predict_test = np.nan_to_num(predict_test,nan=0)
    predict_test = np.nan_to_num(predict_test)
    predict_test = remove_nan(predict_test)
    print(np.any(np.isinf(predict_test['bce_mean'])))
    print(np.any(np.isinf(predict_test['bce_var'])))
    print(np.any(np.isinf(predict_test['bce_waic'])))

    # predict_ood_list = [bae_model.predict(ood_data) for ood_data in ood_data_list]
    predict_ood_list = [bae_model.predict(ood_data) for ood_data in ood_data_list]
    predict_ood_list = [remove_nan(predict_ood) for predict_ood in predict_ood_list]

    #plot reconstruction image of test set
    plot_samples_img(data=predict_test, reshape_size=(28,28), savefile=bae_model.model_name +"-"+"TEST"+"-samples"+".png")

    #evaluate performance curves by comparing against OOD datasets
    for predict_ood,ood_data_name in zip(predict_ood_list,ood_data_names):

        plot_roc_curve(predict_test,predict_ood, title=bae_model.model_name +"-"+ood_data_name, savefile=bae_model.model_name +"-"+ood_data_name+"-AUROC"+".png")
        plot_prc_curve(predict_test,predict_ood, title=bae_model.model_name +"-"+ood_data_name, savefile=bae_model.model_name +"-"+ood_data_name+"-AUPRC"+".png")

        #evaluation
        auroc_list, fpr80_list, metric_names= calc_auroc(predict_test,predict_ood, exclude_keys=exclude_keys)
        auprc_list, metric_names= calc_auprc(predict_test,predict_ood, exclude_keys=exclude_keys)
        plot_samples_img(data=predict_ood, reshape_size=(28,28), savefile=bae_model.model_name +"-"+ood_data_name+"-samples"+".png")
        #save performance results as csv
        save_csv_metrics(train_set_name+"_"+ood_data_name,bae_model.model_name , auroc_list, auprc_list, fpr80_list, metric_names)

    #plot and compare distributions of data per model
    plot_output_distribution(predict_test,*predict_ood_list,legends=["TEST"]+ood_data_names,exclude_keys=dist_exclude_keys, savefile=bae_model.model_name +"-dist"+".png")
    output_means, output_medians, dist_metric_names = calc_variance_dataset(predict_test,*predict_ood_list,legends=["TEST"]+ood_data_names, exclude_keys=["aleatoric_var","waic_se","nll_homo_mean","nll_homo_var","waic_homo","waic_sigma"])
    save_csv_distribution(train_set_name,bae_model.model_name, output_means, output_medians, dist_metric_names)

    homoscedestic_noise = bae_model.get_homoscedestic_noise()
    print(bae_model.model_name +" Homo-noise:"+str(homoscedestic_noise))

    plot_calibration_curve(bae_model, id_data_test)
    # plot_latent(x_test, x_ood, legends=["TEST"]+ood_data_names, model=bae_model, transform_pca=True)
    plot_latent(x_test, x_ood, legends=["TEST"]+ood_data_names, model=bae_model, transform_pca=True)
    plot_train_loss(model=bae_model,savefile=bae_model.model_name+"-loss.png")

# def plot_latent(*datasets, legends=[],model, transform_pca=True, alpha=0.9):
#     fig, (ax_mu,ax_sig) = plt.subplots(1,2)
#
#
#     data_latent_mu_test,data_latent_sig_test = model.predict_latent(datasets[0],transform_pca=False)
#     data_latent_mu_ood,data_latent_sig_ood = model.predict_latent(datasets[0],transform_pca=False)
#
#     pca = PCA(n_components=2)
#     latent_pca_mu = pca.fit_transform(latent_mu)
#
#
#         # ax_mu.scatter(data_latent_mu[:,0],data_latent_mu[:,1],alpha=alpha)
#         # ax_sig.scatter(data_latent_sig[:,0],data_latent_sig[:,1],alpha=alpha)
#
#
#     if len(legends) >0:
#         ax_mu.legend(legends)
#         ax_sig.legend(legends)
#
#     #set titles
#     ax_mu.set_title(model.model_name+" latent mean")
#     ax_sig.set_title(model.model_name+" latent variance")
#
#
#
#
# def plot_train_loss(model):
#     plt.figure()
#     losses = model.losses
#     if len(model.losses)>=5:
#         losses = losses[5:]
#
#     plt.plot(np.arange(1,len(losses)+1),losses)
#     plt.title(model.model_name+" Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.xticks(np.arange(1,len(losses)+1))
#
# def bce_loss_np(y_pred,y_true):
#     return y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)
#
# def mse_loss_np(y_pred,y_true):
#     return (y_pred-y_true)**2
#
#
# # bce_test = bce_loss_np(predict_test['mu'], flatten_np(predict_test['input']))
# # bce_ood = bce_loss_np(predict_ood['mu'], flatten_np(predict_ood['input']))
#
# nll_test = F.binary_cross_entropy(flatten_torch(torch.from_numpy(predict_test['mu'])), flatten_torch(torch.from_numpy(predict_test['input'])),reduction="none")
# nll_test = nll_test.mean(1).detach().cpu().numpy()
# nll_ood = F.binary_cross_entropy(flatten_torch(torch.from_numpy(predict_ood['mu'])), flatten_torch(torch.from_numpy(predict_ood['input'])),reduction="none")
# nll_ood = nll_ood.mean(1).detach().cpu().numpy()
#
# plt.figure()
# plt.hist(nll_test, alpha=0.9)
# plt.hist(nll_ood, alpha=0.9)
#
# bce_test = bae_ensemble.bce_loss_np(predict_test['mu'],flatten_np(predict_test['input']))
# bce_ood = bae_ensemble.bce_loss_np(predict_ood['mu'],flatten_np(predict_ood['input']))
#
#
# bce_test=bce_test.mean(1)
# bce_ood=bce_ood.mean(1)
#
# plt.figure()
# plt.hist(bce_test, alpha=0.9)
# plt.hist(bce_ood, alpha=0.9)
#
#
# mse_test = mse_loss_np(predict_test['mu'], flatten_np(predict_test['input']))
# mse_ood = mse_loss_np(predict_ood['mu'], flatten_np(predict_ood['input']))
#
# mse_test=mse_test.mean(1)
# mse_ood=mse_ood.mean(1)
#
# plt.figure()
# plt.hist(mse_test, alpha=0.9)
# plt.hist(mse_ood, alpha=0.9)
#
