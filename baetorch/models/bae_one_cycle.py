import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.stats as stats

seed_value=123
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
                            weight_decay=1, learning_rate=0.005, learning_rate_sig=0.001,
                            num_samples=5, homoscedestic_mode=homoscedestic_mode)

# ensemble = [create_model(Randomforest) for i in range(5)]

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
# bae_ensemble.fit(train_loader, num_epochs=20, mode="mu")
# bae_ensemble.fit(train_loader, num_epochs=5, mode="sigma", sigma_train="separate")
# bae_ensemble.fit(train_loader, num_epochs=5, mode="sigma", sigma_train="joint")


# bae_ensemble.learning_rate = 0.001
# bae_ensemble.fit_one(x_train, num_epochs=1)

import matplotlib.pyplot as plt
import numpy as np

lr_list = []
train_batch_number = len(train_loader) #num iterations
n=train_batch_number
max_lr=100
init_lr =0.00001
for i in range(n):
    q= (max_lr/init_lr)**(1/n)
    print(q)
    lr_i = init_lr*(q**i)
    lr_list.append(lr_i)

plt.figure()
plt.plot(np.arange(n),lr_list)

mode = "mu"
loss_list = []
for batch_idx, (data, target) in enumerate(train_loader):
    bae_ensemble.learning_rate = lr_list[batch_idx]
    bae_ensemble.set_optimisers(bae_ensemble.autoencoder, mode=mode)
    loss = bae_ensemble.fit_one(x=data,y=data, mode=mode)
    loss_list.append(loss)
    print(loss)

plt.figure()
plt.plot(np.log10(lr_list), loss_list)

#calculate smoothened loss
beta = 0.98
smoothen_loss_list = []
window_size = 5
for loss_i,current_loss in enumerate(loss_list):
    if loss_i>0:
        previous_loss = loss_list[loss_i-1]
        smoothen_loss = beta*previous_loss+(1-beta)*current_loss
        smoothen_loss_list.append(smoothen_loss)

window_size = 10
smoothen_loss_list = []
smoothen_loss_list.append(np.mean(loss_list[0:window_size]))
for loss_i,current_loss in enumerate(loss_list):
    k = 2/(window_size+1)
    if loss_i >=window_size:
        smoothen_loss = (current_loss * k) + smoothen_loss_list[-1]*(1-k)
        # previous_loss = loss_list[loss_i-1]
        # smoothen_loss = beta*previous_loss+(1-beta)*current_loss
        smoothen_loss_list.append(smoothen_loss)


plt.figure()
plt.plot((lr_list)[window_size-1:], smoothen_loss_list)
plt.xscale('log')

#get minimum-maximum lr
# max_multiple = 2
# min_multiple = 10
# minimum_lr_arg = np.argwhere(smoothen_loss_list == np.min(smoothen_loss_list)).flatten()[0]
# minimum_lr = lr_list[minimum_lr_arg]
# minimum_lr = minimum_lr/min_multiple
# minimum_lr_arg= np.argwhere(np.array(lr_list) >= minimum_lr_arg).flatten()[0]
# maximum_lr_arg = np.argwhere(smoothen_loss_list >=smoothen_loss_list[minimum_lr_arg_updated]*max_multiple).flatten()[0]
# maximum_lr_arg = np.argwhere(smoothen_loss_list >=).flatten()[0]

# maximum_lr = smoothen_loss_list[maximum_lr_arg]
# minimum_lr = 0.005
# maximum_lr = 0.005

minimum_lr = 0.00005
maximum_lr = 0.01

# plt.vlines(lr_list[minimum_lr_arg_updated],ymin=0,ymax=10)
# plt.vlines(lr_list[maximum_lr_arg],ymin=0,ymax=10)

#now implement one-cycle training policy
#it should be a triangle shape

#for first half-cycle

total_iterations = len(train_loader)
half_iterations = int(total_iterations/2)
lr_decay = np.abs((maximum_lr-minimum_lr))/half_iterations
lr_rise = lr_decay*np.arange(1,half_iterations)
lr_fall = copy.copy(lr_rise)[::-1]

anni_min_scale = 100
annihilate_iterations = int(total_iterations/anni_min_scale)
lr_anni_decay = np.abs((lr_fall[-1])-(lr_fall[-1]/anni_min_scale))/annihilate_iterations
lr_anni= lr_anni_decay*np.arange(0,annihilate_iterations)[::-1]+(lr_fall[-1]/anni_min_scale)
lr_schedule =np.concatenate((lr_rise,lr_fall,lr_anni))

plt.figure()
plt.plot(np.arange(len(lr_schedule)),lr_schedule)

#now execute one cycle training
bae_ensemble = BAE_Ensemble(autoencoder=autoencoder_3, use_cuda=True, anchored=False,
                            weight_decay=0, learning_rate=0.005, learning_rate_sig=0.001,
                            num_samples=5, homoscedestic_mode=homoscedestic_mode)

#own implementation
# mode = "mu"
# loss_list = []
# lr_schedule_run = []
# current_iteration = 0
# num_epoch = 10
# for epoch in range(num_epoch):
#     current_iteration = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         # bae_ensemble.set_learning_rate(lr_schedule[current_iteration])
#         lr_schedule_run.append(lr_schedule[current_iteration])
#         current_iteration+=1
#         loss = bae_ensemble.fit_one(x=data,y=data, mode=mode)
#         loss_list.append(loss)
#         print(loss)

#using pytorch implementation
mode = "mu"
loss_list = []
lr_schedule_run = []
current_iteration = 0
num_epoch = 10

scheduler = [torch.optim.lr_scheduler.CyclicLR(optimiser, step_size_up=half_iterations, base_lr=minimum_lr, max_lr=maximum_lr, cycle_momentum=False) for optimiser in bae_ensemble.optimisers]
for epoch in range(num_epoch):
    current_iteration = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # bae_ensemble.set_learning_rate(lr_schedule[current_iteration])
        lr_schedule_run.append(lr_schedule[current_iteration])
        current_iteration+=1
        loss = bae_ensemble.fit_one(x=data,y=data, mode=mode)
        loss_list.append(loss)
        print(loss)
        for schedule in scheduler:
            schedule.step()

plt.figure()
plt.plot(loss_list)
print("FINAL LOSS:"+str(np.mean(loss_list[-10:])))



#for each model, evaluate and plot:
bae_models = [bae_ensemble]
id_data_test = test_loader
# ood_data_list = [x_ood]
ood_data_list = [ood_loader]
ood_data_names = ["OOD"]
exclude_keys =[]
dist_exclude_keys = ["aleatoric_var","waic_se","nll_homo_var","waic_homo","waic_sigma"]
train_set_name = "FashionMNIST"


run_test_model(bae_models=bae_models, id_data_test=test_loader, ood_data_list=ood_data_list,
               ood_data_names=ood_data_names, id_data_name=train_set_name)
