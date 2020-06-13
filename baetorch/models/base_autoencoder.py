#Provides 4 options to train the BAE
#1. MCMC
#2. VI
#3. Dropout
#4. Ensembling
#5. VAE (special case of VI?)

#Layer types
#1. Dense
#2. Conv2D
#3. Conv2DTranspose

#Activation layers
#Sigmoid, relu , etc
#Option to configure Last layer

#Parameters of specifying model
#Encoder
#Latent
#Decoder-MU
#Decoder-SIG
#Cluster (TBA)

#Specifying model flow
#1. specify architecture for Conv2D (encoder)
#2. specify architecture for Dense (encoder) #optional
#3. specify architecture for Dense (latent)
#4. specify architecture for Dense (decoder) #optional
#5. specify architecture for Conv2D (decoder)
#since decoder and encoder are symmetrical, end-user probably just need to specify encoder architecture

import torch
import copy
from torch.nn import Parameter
import torch.nn.functional as F
import baetorch.util.dense_util as bnn_utils
from baetorch.util.misc import create_dir
import numpy as np
from baetorch.util.conv2d_util import calc_required_padding, calc_flatten_conv2d_forward_pass, calc_flatten_conv2dtranspose_forward_pass
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.decomposition import PCA
from baetorch.util.misc import parse_activation
from baetorch.util.distributions import CB_Distribution
import math

###TORCH BASE MODULES###
class ConvLayers(torch.nn.Module):
    def __init__(self, input_dim=28, conv_architecture=[1,32,64], conv_kernel=3,
                 conv_stride=1,conv_padding=2, reverse_params=True,
                 mpool_kernel=2,mpool_stride=2, output_padding=[], use_cuda=False, activation="relu",
                 upsampling=False, last_activation="sigmoid", layer_type=[torch.nn.Conv2d, torch.nn.ConvTranspose2d]):
        super(ConvLayers, self).__init__()
        self.layers = []
        self.use_cuda = use_cuda
        self.upsampling = upsampling
        self.conv_architecture = copy.copy(conv_architecture)
        self.conv_kernel,self.conv_stride,self.conv_padding = copy.copy(conv_kernel),copy.copy(conv_stride),copy.copy(conv_padding)
        self.conv_kernel = self.convert_int_to_list(self.conv_kernel,len(self.conv_architecture)-1)
        self.conv_stride = self.convert_int_to_list(self.conv_stride,len(self.conv_architecture)-1)
        self.input_dim = input_dim
        self.activation = activation
        self.last_activation = last_activation

        #forward and deconvolutional layer type
        self.conv2d_layer_type = layer_type[0]
        self.conv2d_trans_layer_type = layer_type[1]

        if len(output_padding) ==0:
            self.conv_padding, self.output_padding = calc_required_padding(input_dim_init=input_dim, kernels=conv_kernel, strides=conv_stride,verbose=True)
        else:
            self.conv_padding = self.convert_int_to_list(self.conv_padding,len(self.conv_architecture)-1)
            self.output_padding = output_padding

        self.last_activation = last_activation

        if self.upsampling and reverse_params:
            self.conv_architecture.reverse()
            self.conv_kernel.reverse()
            self.conv_padding.reverse()
            self.conv_stride.reverse()

        #create sequence of conv2d layers and max pools
        for channel_id,num_channels in enumerate(self.conv_architecture):
            if channel_id != (len(self.conv_architecture)-1):
                in_channels = self.conv_architecture[channel_id]
                out_channels = self.conv_architecture[channel_id+1]

                #activation of last layer
                if channel_id == (len(self.conv_architecture)-2) and self.upsampling:
                    activation = parse_activation(last_activation)
                else:
                    activation = parse_activation(activation)

                #standard convolutional
                if self.upsampling == False:
                    if activation is not None:
                        layer = torch.nn.Sequential(
                            self.conv2d_layer_type(in_channels=in_channels, out_channels=out_channels, kernel_size=self.conv_kernel[channel_id], stride=self.conv_stride[channel_id], padding=self.conv_padding[channel_id]),
                            activation)
                    else:
                        layer = torch.nn.Sequential(
                            self.conv2d_layer_type(in_channels=in_channels, out_channels=out_channels, kernel_size=self.conv_kernel[channel_id], stride=self.conv_stride[channel_id], padding=self.conv_padding[channel_id]))

                #deconvolutional
                else:
                    if activation is not None:
                        layer = torch.nn.Sequential(
                            self.conv2d_trans_layer_type(in_channels=in_channels, out_channels=out_channels, kernel_size=self.conv_kernel[channel_id], stride=self.conv_stride[channel_id], padding=self.conv_padding[channel_id], output_padding=self.output_padding[channel_id]),
                            activation
                            )
                    else:
                        layer = torch.nn.Sequential(
                            self.conv2d_trans_layer_type(in_channels=in_channels, out_channels=out_channels, kernel_size=self.conv_kernel[channel_id], stride=self.conv_stride[channel_id], padding=self.conv_padding[channel_id], output_padding=self.output_padding[channel_id]))

                self.layers.append(layer)


        self.layers = torch.nn.ModuleList(self.layers)

        if self.use_cuda:
            self.layers.cuda()

    def get_input_dimensions(self, flatten=True):
        if flatten:
            return ((self.input_dim**2)*self.conv_architecture[0])
        else:
            return (self.conv_architecture[0],self.input_dim)

    def get_output_dimensions(self,input_dim=None, flatten=True):
        if input_dim is None:
            input_dim = self.input_dim

        if self.upsampling == False:
            return calc_flatten_conv2d_forward_pass(input_dim, channels=self.conv_architecture, strides=self.conv_stride, kernels=self.conv_kernel, paddings=self.conv_padding, flatten=flatten)
        else:
            return calc_flatten_conv2dtranspose_forward_pass(input_dim, channels=self.conv_architecture, strides=self.conv_stride[::-1], kernels=self.conv_kernel[::-1], paddings=self.conv_padding[::-1], output_padding=self.output_padding, flatten=flatten)

    def convert_int_to_list(self,int_param,num_replicate):
        """
        To handle integer passed as param, creates replicate of list
        """
        if isinstance(int_param,int):
            return [int_param]*num_replicate
        else:
            return int_param

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DenseLayers(torch.nn.Module):
    def __init__(self, input_size=1, output_size=1, architecture=["d1","d1"], activation="relu", use_cuda=False, init_log_noise=1e-3, last_activation="none",
                 layer_type=torch.nn.Linear, log_noise_size=1, **kwargs):
        super(DenseLayers, self).__init__()
        self.architecture = architecture
        self.use_cuda = use_cuda
        self.input_size = input_size
        self.output_size = output_size
        self.init_log_noise = init_log_noise
        self.last_activation = last_activation
        self.activation = activation

        #parse architecture string and add
        self.layers = self.init_layers(layer_type)
        self.log_noise_size=log_noise_size
        self.set_log_noise(self.init_log_noise,log_noise_size=log_noise_size)
        self.model_kwargs = kwargs



        #handle activation layers
        if isinstance(activation,str) and activation == "relu":
            self.activation_layer = F.relu
        elif isinstance(activation,str) and activation == "tanh":
            self.activation_layer = torch.tanh
        else:
            self.activation_layer = activation


    def get_input_dimensions(self):
        return self.input_size

    def set_log_noise(self, log_noise,log_noise_size=1):
        if log_noise_size == 0: #log noise is turned off
            self.log_noise = Parameter(torch.FloatTensor([[0.]]),requires_grad=False)
        else:
            self.log_noise = Parameter(torch.FloatTensor([[np.log(log_noise)]*log_noise_size]))

    def init_layers(self, layer_type=torch.nn.Linear, architecture=None, input_size=None,output_size=None, last_activation=None):
        #resort to default input_size
        if input_size is None:
            input_size = self.input_size
        else:
            self.input_size = input_size
        if output_size is None:
            output_size = self.output_size
        else:
            self.output_size = output_size
        if last_activation is None:
            last_activation = self.last_activation
        else:
            self.last_activation = last_activation

        #resort to default architecture
        if architecture is None:
            layers = bnn_utils.parse_architecture_string(input_size,output_size, self.architecture, layer_type=layer_type, last_activation=last_activation)
        else:
            layers = bnn_utils.parse_architecture_string(input_size,output_size, architecture, layer_type=layer_type, last_activation=last_activation)

        if self.use_cuda:
            layers = torch.nn.ModuleList(layers).cuda()
        else:
            layers = torch.nn.ModuleList(layers)
        return layers

    def forward(self,x):
        #apply relu
        for layer_index,layer in enumerate(self.layers):
            x = layer(x)
        return x

def flatten_torch(x):
    x = x.view(x.size()[0], -1)
    return x

def flatten_np(x):
    x = x.reshape(x.shape[0], -1)
    return x

class Flatten(torch.nn.Module):
    def forward(self, x):
        if isinstance(x,tuple):
            y = flatten_torch(x[0])
            return y, x[1:]
        else:
            y = flatten_torch(x)
            return y

class Reshape(torch.nn.Module):
    def __init__(self, size=[]):
        super(Reshape, self).__init__()
        self.size = size

    def forward(self, x):
        if isinstance(x,tuple):
            y = x[0].view(x[0].size()[0], *tuple(self.size))
            return y, x[1:]
        else:
            y = x.view(x.size()[0], *tuple(self.size))
            return y

def infer_decoder(encoder=[],latent_dim=None, last_activation="sigmoid", activation=None):
    decoder=[]
    has_conv_layer = False

    #encoder layers
    for encoder_layer in encoder:
        if isinstance(encoder_layer, ConvLayers):
           #set the activation
           if activation is None:
               activation = encoder_layer.activation

           conv_transpose_inchannels, conv_transpose_input_dim = encoder_layer.get_output_dimensions(flatten=False)[-1]
           decoder_conv = ConvLayers(input_dim=conv_transpose_input_dim, conv_architecture=encoder_layer.conv_architecture,
                                    conv_kernel=encoder_layer.conv_kernel,
                                    conv_stride=encoder_layer.conv_stride,
                                    conv_padding=encoder_layer.conv_padding,
                                    output_padding=encoder_layer.output_padding,
                                    activation=activation,
                                    upsampling=True, last_activation=last_activation)
           decoder.append(decoder_conv)
           has_conv_layer = True

        elif isinstance(encoder_layer, DenseLayers):
            #set the activation
            if activation is None:
                activation = encoder_layer.activation
            dense_architecture = copy.deepcopy(encoder_layer.architecture)
            if latent_dim is None:
                latent_dim = copy.deepcopy(encoder_layer.output_size)

            dense_architecture.reverse()
            decoder_dense= DenseLayers(architecture=dense_architecture,input_size=latent_dim,
                                       output_size=encoder_layer.input_size,
                                       activation=activation, last_activation=last_activation)
            decoder.append(decoder_dense)

    #has convolutional layer, add a reshape layer before the conv layer
    if has_conv_layer:
        decoder.reverse()
        decoder.insert(-1, Reshape((decoder_conv.conv_architecture[0],decoder_conv.input_dim,decoder_conv.input_dim)))

    return torch.nn.Sequential(*decoder)

class Encoder(torch.nn.Sequential):
    def __init__(self, layer_list, input_dim=None):
        super(Encoder, self).__init__(*self.connect_layers(layer_list=layer_list,input_dim=input_dim))

    def connect_layers(self,layer_list, input_dim=None):
        has_conv_layer = False
        for layer in layer_list:
            if isinstance(layer, ConvLayers):
                has_conv_layer = True
        self.has_conv_layer = has_conv_layer

        if has_conv_layer:
            dense_layer = layer_list[-1]
            conv_layer = layer_list[0]

            #reset dense layer inputs to match that of flattened Conv2D Output size
            dense_layer.layers = dense_layer.init_layers(input_size=conv_layer.get_output_dimensions(input_dim)[-1])

            #append flatten layer in between
            return (conv_layer,Flatten(), dense_layer)
        else:
            dense_layer = layer_list[-1]
            return (Flatten(), dense_layer)

    def get_input_dimensions(self):
        for child in self.children():
            if hasattr(child,"get_input_dimensions"):
                return child.get_input_dimensions()

#Autoencoder base class
class Autoencoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Sequential, decoder_mu=None, decoder_sig=None, homoscedestic_mode="none", use_cuda=False):
        super(Autoencoder, self).__init__()
        self.encoder = encoder

        if decoder_mu is None:
            self.decoder_mu = infer_decoder(encoder)
        else:
            self.decoder_mu = decoder_mu

        #decoder sig status
        if decoder_sig is None:
            self.decoder_sig_enabled = False
        else:
            self.decoder_sig_enabled = True
            self.decoder_sig = decoder_sig

        #log noise mode
        self.homoscedestic_mode = homoscedestic_mode
        self.set_log_noise(homoscedestic_mode=self.homoscedestic_mode)

        #set cuda
        self.set_cuda(use_cuda)

    def set_child_cuda(self, child, use_cuda=False):
        if isinstance(child, torch.nn.Sequential):
            for child in child.children():
                child.use_cuda = use_cuda
        else:
            child.use_cuda = use_cuda

    def set_cuda(self, use_cuda=False):

        self.set_child_cuda(self.encoder,use_cuda)
        self.set_child_cuda(self.decoder_mu,use_cuda)

        if self.decoder_sig_enabled:
            self.set_child_cuda(self.decoder_sig,use_cuda)
        self.use_cuda = use_cuda

        if use_cuda:
            self.cuda()
        else:
            self.cpu()

    def set_log_noise(self, homoscedestic_mode=None, init_log_noise=1e-3):
        """
        For homoscedestic regression, sets the dimensions of free parameter `log_noise`.
        We assume three possible cases :
        - "single" : size of 1
        - "every" : size equals to output dimensions
        - "none" : not inferred, and its exact value is set to 1. This is causes the neg log likelihood loss to be equal to MSE.
        """
        if homoscedestic_mode is None:
            homoscedestic_mode = self.homoscedestic_mode

        if homoscedestic_mode== "single":
            self.log_noise = Parameter(torch.FloatTensor([np.log(init_log_noise)]*1))
        elif homoscedestic_mode== "every":
            if isinstance(self.encoder, Encoder):
                log_noise_size = self.encoder.get_input_dimensions()
            else:
                for child in self.encoder.children():
                    if hasattr(child,"get_input_dimensions"):
                        log_noise_size = child.get_input_dimensions()
                        break
            self.log_noise = Parameter(torch.FloatTensor([np.log(init_log_noise)]*log_noise_size))
        elif homoscedestic_mode== "none":
            self.log_noise = Parameter(torch.FloatTensor([[0.]]),requires_grad=False)

    def reset_parameters(self):
        self._reset_nested_parameters(self.encoder)
        self._reset_nested_parameters(self.decoder_mu)
        if self.decoder_sig_enabled:
            self._reset_nested_parameters(self.decoder_sig)
        return self

    def _reset_parameters(self,child_layer):
        if hasattr(child_layer, 'reset_parameters'):
            child_layer.reset_parameters()

    def _reset_nested_parameters(self,network):
        if hasattr(network,"children"):
            for child_1 in network.children():
                for child_2 in child_1.children():
                    self._reset_parameters(child_2)
                    for child_3 in child_2.children():
                        self._reset_parameters(child_3)
                        for child_4 in child_3.children():
                            self._reset_parameters(child_4)
        return network

    def forward(self, x):
        encoded = self.encoder(x)
        decoded_mu = self.decoder_mu(encoded)

        if self.decoder_sig_enabled:
            decoded_sig = self.decoder_sig(encoded)
            return tuple([decoded_mu,decoded_sig])
        else:
            return decoded_mu

###MODEL MANAGER: FIT/PREDICT
#BAE Base class
class BAE_BaseClass():
    def __init__(self, autoencoder=Autoencoder, num_samples=100, anchored=False, weight_decay=0.01,
                 num_epochs=10, verbose=True, use_cuda=False, task="regression", learning_rate=0.01, learning_rate_sig=None,
                 homoscedestic_mode="none", model_type="stochastic", model_name="BAE", scheduler_enabled=False, likelihood="gaussian", denoising_factor=0, output_clamp=(-10,10)):

        #save kwargs
        self.num_samples = num_samples
        self.anchored = anchored
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.anchored = anchored
        self.use_cuda = use_cuda
        self.task = task
        self.learning_rate = learning_rate
        self.mode = "mu"
        self.model_type = model_type
        self.model_name = model_name
        self.losses = []
        self.scheduler_enabled = scheduler_enabled
        self.likelihood = likelihood
        self.denoising_factor = denoising_factor
        self.num_iterations = 1

        #set output clamp
        if output_clamp == (0,0):
            self.output_clamp = False
        else:
            self.output_clamp = output_clamp

        if learning_rate_sig is None:
            self.learning_rate_sig = learning_rate
        else:
            self.learning_rate_sig = learning_rate_sig

        #override homoscedestic mode of autoencoder
        if homoscedestic_mode is not None:
            autoencoder.set_log_noise(homoscedestic_mode=homoscedestic_mode)

        #revert to default
        else:
            homoscedestic_mode = autoencoder.homoscedestic_mode
        self.homoscedestic_mode = homoscedestic_mode

        #init BAE
        self.decoder_sigma_enabled = autoencoder.decoder_sig_enabled
        self.autoencoder = self.init_autoencoder(autoencoder)
        self.set_optimisers(self.autoencoder, self.mode)

    def save_model_state(self, filename=None, folder_path="torch_model/"):
        create_dir(folder_path)
        if filename is None:
            temp = True
        if self.model_type == "list":
            for model_i,autoencoder in enumerate(self.autoencoder):
                if temp:
                    torch_filename = self.model_name+"_"+str(model_i)+".pt"
                    torch_filename = "temp_"+torch_filename
                else:
                    torch_filename = temp
                torch.save(autoencoder.state_dict(), folder_path+torch_filename)

        else: #stochastic model
            if temp:
                torch_filename = self.model_name+".pt"
                torch_filename = "temp_"+torch_filename
            else:
                torch_filename = temp
            torch.save(self.autoencoder.state_dict(), folder_path+torch_filename)

    def load_model_state(self, filename=None, folder_path="torch_model/"):
        create_dir(folder_path)
        if filename is None:
            temp = True
        if self.model_type == "list":
            for model_i,autoencoder in enumerate(self.autoencoder):
                if temp:
                    torch_filename = self.model_name+"_"+str(model_i)+".pt"
                    torch_filename = "temp_"+torch_filename
                else:
                    torch_filename = temp
                self.autoencoder[model_i].load_state_dict(torch.load(folder_path+torch_filename))
        else: #stochastic model
            if temp:
                torch_filename = self.model_name+".pt"
                torch_filename = "temp_"+torch_filename
            else:
                torch_filename = temp
            self.autoencoder.load_state_dict(torch.load(folder_path+torch_filename))

    def _get_homoscedestic_noise(self, model_type="list"):
        """
        Internal method to access the autoenocder's free log noise parameters, depending on the type of model.
        For example, this depends on whether the model is a list or a stochastic model
        Developers should override this to provide custom function to access the autoencoder's parameters

        This is similar in concept to the method `_predict_samples`
        """
        if model_type == "list":
            log_noise = np.array([model.log_noise.detach().cpu().numpy() for model in self.autoencoder])
            log_noise_mean = np.exp(log_noise).mean(0)
            log_noise_var = np.exp(log_noise).var(0)
            return (log_noise_mean, log_noise_var)
        elif model_type == "stochastic":
            log_noise = np.exp(self.autoencoder.log_noise.detach().cpu().numpy())
            return (log_noise, np.zeros_like(log_noise))

    def get_homoscedestic_noise(self, return_mean = True):
        """
        Returns mean and variance (if available) of the estimated homoscedestic noise
        """
        if return_mean:
            homo_noise = self._get_homoscedestic_noise(model_type=self.model_type)
            return homo_noise[0].mean()
        else:
            return self._get_homoscedestic_noise(model_type=self.model_type)

    def init_autoencoder(self, autoencoder: Autoencoder):
        """
        This function is called on the autoencoder to re-initialise it with the parameters of this BAE model manager.
        """

        model = self.convert_autoencoder(autoencoder)
        model.set_log_noise(homoscedestic_mode=self.homoscedestic_mode)
        if self.anchored:
            model = self.init_anchored_weight(model)

        model.set_cuda(self.use_cuda)

        return model

    def convert_autoencoder(self, autoencoder: Autoencoder):
        """
        Augment the autoencoder architecture to enable Bayesian approximation.
        Returns the forward augmented model.

        Developers should override this to convert the given autoencoder into a BAE.
        """
        return copy.deepcopy(autoencoder)

    def get_optimisers_list(self, autoencoder: Autoencoder, mode="mu", sigma_train="separate"):
        optimiser_list = []
        if mode =="sigma":
            if autoencoder.decoder_sig_enabled:
                optimiser_list.append({'params':autoencoder.decoder_sig.parameters(),'lr':self.learning_rate_sig})
                if sigma_train == "joint": #for joint training
                    optimiser_list.append({'params':autoencoder.encoder.parameters()})
                    optimiser_list.append({'params':autoencoder.decoder_mu.parameters()})
                    optimiser_list.append({'params':autoencoder.log_noise})
        else:
            optimiser_list.append({'params':autoencoder.encoder.parameters()})
            optimiser_list.append({'params':autoencoder.decoder_mu.parameters()})
            optimiser_list.append({'params':autoencoder.log_noise})

        return optimiser_list

    def reset_parameters(self):
        if self.model_type == "list":
            for autoencoder in self.autoencoder:
                autoencoder.reset_parameters()
        else:
            self.autoencoder.reset_parameters()

    def get_optimisers(self, autoencoder: Autoencoder, mode="mu", sigma_train="separate"):
        optimiser_list = self.get_optimisers_list(autoencoder, mode=mode, sigma_train=sigma_train)
        return torch.optim.Adam(optimiser_list, lr=self.learning_rate)
        # return torch.optim.SGD(optimiser_list, lr=self.learning_rate)

    def set_optimisers(self, autoencoder: Autoencoder, mode="mu", sigma_train="separate"):
        if self.model_type == "list":
            self.optimisers = [self.get_optimisers(model, mode=mode,sigma_train=sigma_train) for model in self.autoencoder]
        else:
            self.optimisers = self.get_optimisers(autoencoder, mode=mode,sigma_train=sigma_train)
        return self.optimisers

    def init_scheduler(self, half_iterations=None, min_lr=None, max_lr=None):
        #resort to internal stored values for scheduler
        if half_iterations is None or min_lr is None or max_lr is None:
            half_iterations = self.half_iterations
            min_lr = self.min_lr
            max_lr = self.max_lr
        else:
            self.half_iterations = half_iterations
            self.min_lr = min_lr
            self.max_lr = max_lr

        #handle model type to access optimiser hierarchy
        if self.model_type == "list":
            self.scheduler = [torch.optim.lr_scheduler.CyclicLR(optimiser, step_size_up=half_iterations,
                                                           base_lr=min_lr, max_lr=max_lr,
                                                           cycle_momentum=False) for optimiser in self.optimisers]
        else:
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimisers, step_size_up=half_iterations, base_lr=min_lr, max_lr=max_lr, cycle_momentum=False)
        self.scheduler_enabled = True

        #set init learning rate
        self.learning_rate = self.min_lr
        self.learning_rate_sig = self.min_lr
        return self.scheduler

    def set_learning_rate(self, learning_rate=None):
        #update internal learning rate
        if learning_rate is None:
            learning_rate = self.learning_rate
        else:
            self.learning_rate = learning_rate

        #handle access to optimiser parameters
        if self.model_type =="list":
            for optimiser in self.optimisers:
                for group in optimiser.param_groups:
                    group['lr'] = learning_rate

        if self.model_type =="stochastic":
            for group in self.optimisers.param_groups:
                group['lr'] = learning_rate

    def zero_optimisers(self):
        if self.model_type == "list":
            for optimiser in self.optimisers:
                optimiser.zero_grad()
        else:
            self.optimisers.zero_grad()

    def step_optimisers(self):
        if self.model_type == "list":
            for optimiser in self.optimisers:
                optimiser.step()
        else:
            self.optimisers.step()

    def step_scheduler(self):
        if self.model_type == "list":
            for scheduler in self.scheduler:
                scheduler.step()
        else:
            self.scheduler.step()

    def fit_one(self, x, y=None,mode="mu"):
        """
        Template for vanilla fitting, developers are very likely to override this to provide custom fitting functions.
        """
        #extract input and output size from data
        #and convert into tensor, if not already

        #if denoising is enabled
        if self.denoising_factor:
            y =copy.deepcopy(x)
            x =x + self.denoising_factor * torch.randn(*x.shape)
            x = torch.clamp(x, 0., 1.)

        if y is None:
            y = x
        x,y = self.convert_tensor(x,y)


        #train for n epochs
        loss = self.criterion(autoencoder=self.autoencoder, x=x, y=y, mode=mode)

        #backpropagate
        self.zero_optimisers()
        loss.backward()
        self.step_optimisers()

        #if scheduler is enabled, update it
        if self.scheduler_enabled:
            self.step_scheduler()

        return loss.item()

    def print_loss(self, epoch,loss):
        if self.verbose:
            print("LOSS #{}:{}".format(epoch,loss))

    def fit(self, x,y=None, mode="mu", num_epochs=None, sigma_train="separate"):
        """
        Overarching fitting function, to handle pytorch train loader or tensors
        """
        self.set_optimisers(self.autoencoder, mode=mode,sigma_train=sigma_train)

        if self.scheduler_enabled:
            self.init_scheduler()

        if num_epochs is None:
            num_epochs = self.num_epochs

        #handle train loader
        if isinstance(x, torch.utils.data.dataloader.DataLoader):
            #save the number of iterations for KL weighting later
            self.num_iterations = len(x)

            for epoch in tqdm(range(num_epochs)):
                temp_loss = []
                for batch_idx, (data, target) in enumerate(x):
                    loss = self.fit_one(x=data,y=data, mode=mode)
                    temp_loss.append(loss)
                    self.losses.append(np.mean(temp_loss))
                self.print_loss(epoch,self.losses[-1])

        else:
            for epoch in tqdm(range(num_epochs)):
                loss = self.fit_one(x=x,y=y, mode=mode)
                self.losses.append(loss)
                self.print_loss(epoch,self.losses[-1])
        return self

    def nll(self, autoencoder: Autoencoder, x, y=None, mode="sigma"):
        #likelihood
        if y is None:
            y = x
        if self.decoder_sigma_enabled:
            y_pred_mu, y_pred_sig = autoencoder(x)
        if mode=="sigma":
            nll = self._nll(flatten_torch(y_pred_mu), flatten_torch(y), flatten_torch(y_pred_sig))
        elif mode =="mu":
            y_pred_mu = autoencoder(x)
            nll = self._nll(flatten_torch(y_pred_mu), flatten_torch(y), autoencoder.log_noise)
        return nll

    def _nll(self, y_pred_mu, y, y_pred_sig=None):
        if self.likelihood == "gaussian":
            nll = -self.log_gaussian_loss_logsigma_torch(flatten_torch(y_pred_mu), flatten_torch(y), y_pred_sig)
        elif self.likelihood == "laplace":
            nll = torch.abs(flatten_torch(y_pred_mu)- flatten_torch(y))
        elif self.likelihood == "bernoulli":
            nll = F.binary_cross_entropy(flatten_torch(y_pred_mu), flatten_torch(y), reduction="none")
        elif self.likelihood == "cbernoulli":
            nll = self.log_cbernoulli_loss_torch(flatten_torch(y_pred_mu), flatten_torch(y))
        return nll

    def log_cbernoulli_loss_torch(self, y_pred_mu, y_true):
        if hasattr(self, "cb") == False:
            self.cb = CB_Distribution()
        nll_cb = self.cb.log_cbernoulli_loss_torch(y_pred_mu,y_true)
        return nll_cb

    def log_cbernoulli_loss_np(self, y_pred_mu, y_true):
        if hasattr(self, "cb") == False:
            self.cb = CB_Distribution()
        return self.cb.log_cbernoulli_loss_np(torch.from_numpy(y_pred_mu),torch.from_numpy(y_true))

    def criterion(self, autoencoder: Autoencoder, x,y=None, mode="sigma"):
        #likelihood
        nll = self.nll(autoencoder,x,y,mode)
        nll = nll.mean()

        #prior loss
        prior_loss = self.log_prior_loss(model=autoencoder)
        prior_loss = prior_loss.mean()
        prior_loss *= self.weight_decay

        return nll + prior_loss

    def _predict_samples(self,x, model_type=0):
        """
        Internal method, to be overriden by developer with how predictive samples should
        be collected.

        Here we assume either the self.autoencoder is a list of models, or a stochastic model.

        Reason why we don't convert all into numpy array, is because each model output can be of different dimensions.
        For example, when we have a clustering output which is different from the decoder_mu dimensions.

        Parameters
        ----------
        x : numpy or torch.Tensor
            Input data for model prediction

        model_type : "stochastic" or "list"
            Stochastic mode assumes we pass the data through the self.autoencoder for self.num_samples times
            In list mode, we assume the self.autoencoder consists of list of autoencoder models

        Returns
        -------
        y_preds : list
            List of raw outputs of the models

        """
        #revert to default
        if model_type == 0:
            model_type = self.model_type

        #handle different model types
        if model_type == "list":
            y_preds = [list(self._calc_output_single(model,x)) for model in self.autoencoder]
        elif model_type == "stochastic":
            y_preds = [list(self._calc_output_single(self.autoencoder,x)) for i in range(self.num_samples)]

        return y_preds

    def predict_dataloader(self, dataloader: torch.utils.data.dataloader.DataLoader,exclude_keys: list =[]):
        """
        Accumulate results from each test batch, instead of calculating all at one go.
        """
        final_results ={}

        for batch_idx, (data, target) in tqdm(enumerate(dataloader)):
            #predict new batch of results
            next_batch_result = self._predict(data,exclude_keys)

            #populate for first time
            if batch_idx ==0:
                final_results.update(next_batch_result)
            #append for subsequent batches
            else:
                for key in final_results.keys():
                    final_results[key] = np.concatenate((final_results[key],next_batch_result[key]),axis=0)
        return final_results


    def predict_samples(self,x):
        """
        Returns raw samples from each forward pass of the AE models
        """
        x = self.convert_tensor(x)
        y_preds = self._predict_samples(x)

        y_preds = np.array(y_preds)
        return y_preds

    def predict(self,x,exclude_keys=[]):
        """
        Handles various data types including torch dataloader

        For actual outputs of predictions, see `_predict(x)`

        """
        if isinstance(x, torch.utils.data.dataloader.DataLoader):
            return self.predict_dataloader(x,exclude_keys)
        else: #numpy array or tensor
            return self._predict(x,exclude_keys)

    def _predict(self,x, exclude_keys : list =[]):
        """
        Computes mean and variances of the BAE model outputs

        The outputs can be more than necessary, depending on the context.
        Users will need to filter which ones to include and exclude via keys

        Returns
        -------
        results : dict
        """

        #get raw samples from each model forward pass
        raw_samples = self.predict_samples(x)

        #return also the input
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        #calculate mean and variance of model predictions
        mean_samples = raw_samples.mean(0)
        var_samples = raw_samples.var(0)

        #compute statistics of model outputs
        y_mu_mean, y_sigma_mean, se_mean,bce_mean,cbce_mean, nll_homo_mean,nll_sigma_mean = mean_samples
        y_mu_var, y_sigma_var, se_var,bce_var,cbce_var, nll_homo_var,nll_sigma_var= var_samples

        #total uncertainty = epistemic+aleatoric
        if self.decoder_sigma_enabled:
            total_unc = y_sigma_mean+y_mu_var
        elif self.homoscedestic_mode != "none":
            total_unc = y_sigma_mean+self.get_homoscedestic_noise(return_mean=False)[0]
        else:
            total_unc = y_mu_var

        #calculate waic
        waic_se = se_mean+se_var
        waic_homo = nll_homo_mean+nll_homo_var
        waic_nll = nll_sigma_mean+nll_sigma_var

        #bce loss
        waic_bce = bce_mean+bce_var
        waic_cbce = cbce_mean+cbce_var

        #filter out unnecessary outputs based on model settings of computing aleatoric uncertainty
        if self.decoder_sigma_enabled == False:
            exclude_keys = exclude_keys+["aleatoric","aleatoric_var", "nll_sigma_var", "nll_sigma_mean", "nll_sigma_waic"]
        if self.homoscedestic_mode == "none":
            exclude_keys = exclude_keys+["nll_homo_mean", "nll_homo_var", "nll_homo_waic"]
        if self.homoscedestic_mode == "none" and self.decoder_sigma_enabled == False:
            exclude_keys = exclude_keys+["total_unc"]

        return_dict= {"input":x,"mu":y_mu_mean, "epistemic":y_mu_var,
                "aleatoric":y_sigma_mean,"aleatoric_var":y_sigma_var,
                "total_unc":total_unc, "bce_mean":bce_mean, "bce_var":bce_var, "bce_waic":waic_bce,
                "cbce_mean":cbce_mean, "cbce_var":cbce_var, "cbce_waic":waic_cbce,
                "se_mean":se_mean, "se_var":se_var, "se_waic":waic_se,
                "nll_homo_mean":nll_homo_mean, "nll_homo_var":nll_homo_var, "nll_homo_waic":waic_homo,
                "nll_sigma_mean":nll_sigma_mean, "nll_sigma_var":nll_sigma_var, "nll_sigma_waic":waic_nll
                }

        return_dict = {filtered_key:value for filtered_key,value in return_dict.items() if filtered_key not in exclude_keys}
        return return_dict

    def bce_loss_np(self,y_pred,y_true):
        bce = -(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
        bce = np.nan_to_num(bce, nan=0,posinf=100,neginf=-100)
        return bce

    def cbce_loss_np(self,y_pred,y_true):
        cbce = self.log_cbernoulli_loss_np(y_pred,y_true)
        cbce = np.nan_to_num(cbce, nan=0,posinf=100,neginf=-100)
        return cbce

    def _get_mu_sigma_single(self, autoencoder, x):
        if self.decoder_sigma_enabled:
            y_mu, y_sigma = autoencoder(x)
        else:
            y_mu = autoencoder(x)
            y_sigma = torch.ones_like(y_mu)

        #convert to numpy
        y_mu = y_mu.detach().cpu().numpy()
        y_sigma = y_sigma.detach().cpu().numpy()
        log_noise = autoencoder.log_noise.detach().cpu().numpy()

        return flatten_np(y_mu), flatten_np(y_sigma), log_noise

    def _calc_output_single(self, autoencoder, x):
        #per sample
        y_mu, y_sigma, log_noise = self._get_mu_sigma_single(autoencoder, x)

        #clamp it to min max
        y_mu = np.clip(y_mu, a_min=self.output_clamp[0],a_max=self.output_clamp[1])

        x = flatten_np(x.detach().cpu().numpy())
        se = (y_mu-x)**2
        bce = self.bce_loss_np(y_mu,x)
        cbce = self.cbce_loss_np(y_mu,x)
        nll_homo = -self.log_gaussian_loss_logsigma_np(y_mu,x,log_noise)
        nll_sigma = -self.log_gaussian_loss_logsigma_np(y_mu,x,y_sigma)

        return y_mu, np.exp(y_sigma), se,bce,cbce, nll_homo,nll_sigma

    def log_gaussian_loss_logsigma_torch(self, y_pred, y_true, log_sigma):
        log_likelihood = (-((y_true - y_pred)**2)*torch.exp(-log_sigma)*0.5)-(0.5*log_sigma)
        return log_likelihood

    def log_gaussian_loss_sigma_2_torch(self, y_pred, y_true, sigma_2):
        log_likelihood = (-((y_true - y_pred)**2)/(2*sigma_2))-(0.5*torch.log(sigma_2))
        return log_likelihood

    def log_gaussian_loss_logsigma_np(self, y_pred, y_true, log_sigma):
        log_likelihood = (-((y_true - y_pred)**2)*np.exp(-log_sigma)*0.5)-(0.5*log_sigma)
        return log_likelihood

    def log_gaussian_loss_sigma_2_np(self, y_pred, y_true, sigma_2):
        log_likelihood = (-((y_true - y_pred)**2)/(2*sigma_2))-(0.5*np.log(sigma_2))
        return log_likelihood

    def log_prior_loss(self, model, mu=torch.Tensor([0.]), L=2):
        #prior 0 ,1
        if self.anchored:
            mu = model.anchored_prior

        if self.use_cuda:
            mu=mu.cuda()

        weights = torch.cat([parameter.flatten() for parameter in model.parameters()])
        prior_loss = ((weights - mu)**L)
        return prior_loss

    def get_anchored_weight(self, model):
        model_weights = torch.cat([parameter.flatten() for parameter in model.parameters()])
        anchored_weights = torch.ones_like(model_weights)*model_weights.detach()
        return anchored_weights

    def init_anchored_weight(self, model):
        model.anchored_prior = self.get_anchored_weight(model)
        return model

    def convert_tensor(self,x,y=None):
        if "Tensor" not in type(x).__name__:
            x = Variable(torch.from_numpy(x).float())
        if y is not None:
            if "Tensor" not in type(y).__name__:
                if self.task == "classification":
                    y = Variable(torch.from_numpy(y).long())
                else:
                    y = Variable(torch.from_numpy(y).float())

        #use cuda
        if self.use_cuda:
            if y is None:
                return x.cuda()
            else:
                return x.cuda(),y.cuda()
        else:
            if y is None:
                return x
            else:
                return x,y

    def predict_latent_samples(self,x):
        """
        Function for model to pass a forward to get the samples of latent values
        """
        #handle different model types
        if self.model_type == "list":
            latent_samples = torch.stack([self._forward_latent_single(model,x) for model in self.autoencoder])
        elif self.model_type == "stochastic":
            latent_samples = torch.stack([self._forward_latent_single(self.autoencoder,x) for i in range(self.num_samples)])
        return latent_samples

    def _forward_latent_single(self,model,x):
        return model.encoder(x)

    def predict_latent(self, x, transform_pca=True):
        """
        Since BAE is probabilistic, we can obtain mean and variance of the latent dimensions for each data
        """
        x = self.convert_tensor(x)
        latent_data = self.predict_latent_samples(x)
        latent_mu = latent_data.mean(0).detach().cpu().numpy()
        latent_sigma = latent_data.var(0).detach().cpu().numpy()

        #transform pca
        if transform_pca:
            pca = PCA(n_components=2)
            latent_pca_mu = pca.fit_transform(latent_mu)
            latent_pca_sig = pca.fit_transform(latent_sigma)
            return latent_pca_mu,latent_pca_sig
        else:
            return latent_mu, latent_sigma

    def set_cuda(self, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.use_cuda
        else:
            self.use_cuda = use_cuda

        if self.model_type == "stochastic":
            self.autoencoder.set_cuda(use_cuda)
        else:
            for model in self.autoencoder:
                model.set_cuda(use_cuda)
