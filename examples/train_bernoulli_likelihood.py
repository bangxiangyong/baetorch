#example of how to train using Bernoulli likelihood instead of Gaussian (default)
#it is specified in the `likelihood` parameter when instantiating the BAE model of choice

from baetorch.models.base_autoencoder import *
from torchvision import datasets, transforms
from baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.plotting import *
from baetorch.test_suite import run_test_model
from baetorch.lr_range_finder import run_auto_lr_range

from baetorch.util.seed import bae_set_seed

#set cuda if available
use_cuda = torch.cuda.is_available()


#set seed for reproduciliblity
bae_set_seed(100)

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


#model architecture
latent_dim = 10
input_dim = 28
input_channel = 1

conv_architecture=[input_channel,12,24,48]

#specify encoder
#with convolutional layers and hidden dense layer
encoder = Encoder([ConvLayers(input_dim=input_dim,conv_architecture=conv_architecture, conv_kernel=[4,4,4], conv_stride=[2,2,2], last_activation="sigmoid"),
           DenseLayers(architecture=[],output_size=latent_dim)])

#specify decoder-mu
decoder_mu = infer_decoder(encoder,last_activation="sigmoid") #symmetrical to encoder

#combine them into autoencoder
autoencoder = Autoencoder(encoder, decoder_mu)

#convert into BAE-Ensemble
bae_ensemble = BAE_Ensemble(autoencoder=autoencoder, use_cuda=use_cuda,
                            anchored=True, weight_decay=1,
                            num_samples=5, likelihood="bernoulli")

#train mu network
run_auto_lr_range(train_loader, bae_ensemble)
bae_ensemble.fit(train_loader,num_epochs=1)

#for each model, evaluate and plot:
bae_models = [bae_ensemble]
id_data_test = test_loader
ood_data_list = [ood_loader]
train_set_name = "FashionMNIST"

#run evaluation test of model on ood data set
run_test_model(bae_models=bae_models, id_data_test=test_loader, ood_data_list=ood_data_list, id_data_name=train_set_name, output_reshape_size=(28, 28))
