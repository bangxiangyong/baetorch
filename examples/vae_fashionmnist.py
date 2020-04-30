from baetorch.models.base_autoencoder import *
from torchvision import datasets, transforms
from baetorch.models.bae_vi import VAE
from baetorch.plotting import *
from baetorch.test_suite import run_test_model
from baetorch.lr_range_finder import run_auto_lr_range
from baetorch.util.misc import get_sample_dataloader
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
latent_dim = 30
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

#convert into VAE
bae_model = VAE(autoencoder=autoencoder,
                num_train_samples=5,
                num_samples=50, #during prediction only
                use_cuda=use_cuda,
                weight_decay=0.01)

#train mu network
run_auto_lr_range(train_loader, bae_model)
bae_model.fit(train_loader,num_epochs=1)

#for each model, evaluate and plot:
bae_models = [bae_model]
id_data_test = test_loader
ood_data_list = [ood_loader]
train_set_name = "FashionMNIST"

#run evaluation test of model on ood data set
run_test_model(bae_models=bae_models, id_data_test=test_loader, ood_data_list=ood_data_list, id_data_name=train_set_name, output_reshape_size=(28, 28))


#experimental here
x_test = get_sample_dataloader(test_loader)[0].cuda()
x_ood = get_sample_dataloader(ood_loader)[0].cuda()
test_latent = bae_model.predict_latent(x_test, transform_pca=False)
ood_latent = bae_model.predict_latent(x_ood, transform_pca=False)

plt.figure()
plt.boxplot([test_latent[0].mean(1), ood_latent[0].mean(1)])

plt.figure()
plt.boxplot([test_latent[1].mean(1), ood_latent[1].mean(1)])

# plt.figure()
# plt.boxplot([test_latent_data[0][:,i] for i in range(10)])
#
# plt.figure()
# plt.boxplot([ood_latent_data[0][:,i] for i in range(10)])

#plot signature of latent space
mean_index = 0
var_index = 1
selected_index = var_index
x_range = np.arange(1,latent_dim+1)

# test_latent_medians = np.median(sorted_test_latent,axis=0)
# arg_median_sort=np.argsort(test_latent_medians)
plt.figure()
for i in range(100):
    plt.plot(x_range, test_latent[mean_index][i], alpha=0.25, c="blue")
    plt.plot(x_range, ood_latent[mean_index][i], alpha=0.25, c="red")
plt.xticks(x_range)
plt.ylabel("Value")
plt.xlabel("Mean Latent dimension")
plt.figure()
for i in range(100):
    plt.plot(x_range, test_latent[var_index][i], alpha=0.25, c="blue")
    plt.plot(x_range, ood_latent[var_index][i], alpha=0.25, c="red")
plt.xticks(x_range)
plt.ylabel("Value")
plt.xlabel("Var Latent dimension")

#decode
# latent_sample = bae_model.autoencoder.latent_layer_vi(torch.from_numpy(test_latent_data[0]).cuda(),torch.from_numpy(test_latent_data[1]).cuda()*-100)[0]
# decoded = bae_model.autoencoder.decoder_mu(latent_sample).detach().cpu().numpy()
#
# plt.figure()
# plt.imshow(decoded[55].reshape(28,28))


num_draw_samples = 25
draw_latent_samples = torch.empty((num_draw_samples,latent_dim)).normal_(mean=0,std=0.5).cuda()
draw_decoded = bae_model.autoencoder.decoder_mu(draw_latent_samples).detach().cpu().numpy()

fig, axes=plt.subplots(int(np.sqrt(num_draw_samples)),int(np.sqrt(num_draw_samples)))
axes = axes.flatten()
for ax_id,ax in enumerate(axes):
    if ax_id < num_draw_samples:
        ax.imshow(draw_decoded[ax_id].reshape(28,28))



