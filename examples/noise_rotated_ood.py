#We can evaluate the BAE on noisy and rotated inputs, to see how they behave
#Specifically, we observe evaluation using Bernoulli likelihood (BCE) is more sensitive to noise
#This can be beneficial if the OOD is expected to be noisy
#Based on MSE however, the noise (of small scale) are
#attenuated and both MSE on the noisy and test images yield the same MSE.
#Hence, we can say the MSE is less susceptible to noise.

from baetorch.models.base_autoencoder import *
from torchvision import datasets, transforms
from baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.models.bae_mcdropout import BAE_MCDropout
from baetorch.plotting import *
from baetorch.test_suite import run_test_model
from baetorch.lr_range_finder import run_auto_lr_range
from baetorch.util.seed import bae_set_seed
from baetorch.util.misc import AddNoise

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

#in addition to MNIST as OOD, we can evaluate the BAE on surrogate OOD i.e Rotated or Noisy data
rotated_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data-fashion-mnist', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.RandomRotation([89,90]),
                       transforms.ToTensor(),
                   ])
                   ), batch_size=test_samples, shuffle=True)

noise_factor = 0.05
noisy_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data-fashion-mnist', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       AddNoise(noise_factor)
                   ])
                   ), batch_size=test_samples, shuffle=True)


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

#convert into BAE
bae_model = BAE_MCDropout(autoencoder=autoencoder, dropout_p=0.2,
                              num_train_samples=5, num_samples=50, use_cuda=use_cuda)

#train mu network
run_auto_lr_range(train_loader, bae_model)
bae_model.fit(train_loader,num_epochs=2)

#for each model, evaluate and plot:
bae_models = [bae_model]
id_data_test = test_loader
ood_data_list = [ood_loader,noisy_loader,rotated_loader]
ood_names =["OOD-MNIST","NOISY","ROTATED"]
train_set_name = "FashionMNIST"

#run evaluation test of model on ood data set
run_test_model(bae_models=bae_models, id_data_test=test_loader,
               ood_data_names=ood_names, ood_data_list=ood_data_list,
               id_data_name=train_set_name, output_reshape_size=(28, 28))

