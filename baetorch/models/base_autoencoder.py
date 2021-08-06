# Provides 4 options to train the BAE
# 1. MCMC
# 2. VI
# 3. Dropout
# 4. Ensembling
# 5. VAE (special case of VI?)

# Layer types
# 1. Dense
# 2. Conv2D
# 3. Conv2DTranspose

# Activation layers
# Sigmoid, relu , etc
# Option to configure Last layer

# Parameters of specifying model
# Encoder
# Latent
# Decoder-MU
# Decoder-SIG
# Cluster (TBA)

# Specifying model flow
# 1. specify architecture for Conv2D (encoder)
# 2. specify architecture for Dense (encoder) #optional
# 3. specify architecture for Dense (latent)
# 4. specify architecture for Dense (decoder) #optional
# 5. specify architecture for Conv2D (decoder)
# since decoder and encoder are symmetrical, end-user probably just need to specify encoder architecture
import collections
import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.autograd import Variable
from torch.nn import Parameter
from tqdm import tqdm

from ..models.base_layer import (
    ConvLayers,
    Conv2DLayers,
    Conv1DLayers,
    DenseLayers,
    Reshape,
    Flatten,
    flatten_torch,
    flatten_np,
)
from ..models.cholesky_layer import CholLayer
from ..util.distributions import CB_Distribution, TruncatedGaussian
from ..util.misc import create_dir


def infer_decoder(
    encoder=[],
    latent_dim=None,
    last_activation="sigmoid",
    activation=None,
    bias=False,
    norm=True,
    last_norm=True,
):
    decoder = []
    has_conv_layer = False

    # encoder layers
    for encoder_layer in encoder:
        if isinstance(encoder_layer, ConvLayers):
            # set the activation
            if activation is None:
                activation = encoder_layer.activation

            (
                conv_transpose_inchannels,
                conv_transpose_input_dim,
            ) = encoder_layer.get_output_dimensions(flatten=False)[-1]
            decoder_conv_parameters = {
                "input_dim": conv_transpose_input_dim,
                "conv_architecture": encoder_layer.conv_architecture,
                "conv_kernel": encoder_layer.conv_kernel,
                "conv_stride": encoder_layer.conv_stride,
                "conv_padding": encoder_layer.conv_padding,
                "output_padding": encoder_layer.output_padding,
                "activation": activation,
                "upsampling": True,
                "last_activation": last_activation,
                "bias": bias,
                "norm": norm,
                "last_norm": last_norm,
            }

            # handle either conv1d or conv2d layers
            if encoder_layer.conv_dim == 2:
                decoder_conv = Conv2DLayers(**decoder_conv_parameters)
            else:
                decoder_conv = Conv1DLayers(**decoder_conv_parameters)
            decoder.append(decoder_conv)
            has_conv_layer = True

        elif isinstance(encoder_layer, DenseLayers):
            # set the activation
            if activation is None:
                activation = encoder_layer.activation
            dense_architecture = copy.deepcopy(encoder_layer.architecture)
            if latent_dim is None:
                latent_dim = copy.deepcopy(encoder_layer.output_size)

            dense_architecture.reverse()
            if has_conv_layer:
                decoder_dense = DenseLayers(
                    architecture=dense_architecture,
                    input_size=latent_dim,
                    output_size=encoder_layer.input_size,
                    activation=activation,
                    last_activation=activation,
                    last_norm=norm,
                )
            else:
                decoder_dense = DenseLayers(
                    architecture=dense_architecture,
                    input_size=latent_dim,
                    output_size=encoder_layer.input_size,
                    activation=activation,
                    last_activation=last_activation,
                    last_norm=last_norm,
                )
            decoder.append(decoder_dense)

    # has convolutional layer, add a reshape layer before the conv layer
    if has_conv_layer:
        decoder.reverse()
        if isinstance(decoder_conv.input_dim, tuple):
            decoder.insert(
                -1,
                Reshape((decoder_conv.conv_architecture[0], *decoder_conv.input_dim)),
            )
        elif isinstance(decoder_conv.input_dim, int):
            decoder.insert(
                -1, Reshape((decoder_conv.conv_architecture[0], decoder_conv.input_dim))
            )

    return torch.nn.Sequential(*decoder)


class Encoder(torch.nn.Sequential):
    def __init__(self, layer_list, input_dim=None):
        super(Encoder, self).__init__(
            *self.connect_layers(layer_list=layer_list, input_dim=input_dim)
        )

    def connect_layers(self, layer_list, input_dim=None):
        has_conv_layer = False
        has_dense_layer = False
        for layer in layer_list:
            if isinstance(layer, ConvLayers):
                has_conv_layer = True
            if isinstance(layer, DenseLayers):
                has_dense_layer = True
        self.has_conv_layer = has_conv_layer
        self.has_dense_layer = has_dense_layer

        if has_dense_layer:
            dense_layer = layer_list[-1]
            self.latent_dim = dense_layer.output_size

        if has_conv_layer:

            conv_layer = layer_list[0]

            # reset dense layer inputs to match that of flattened Conv2D Output size
            if has_dense_layer:
                dense_layer.layers = dense_layer.init_layers(
                    input_size=conv_layer.get_output_dimensions(input_dim)[-1]
                )

                # append flatten layer in between
                return (conv_layer, Flatten(), dense_layer)
            else:
                return tuple([conv_layer])
        else:
            return (
                Flatten(),
                dense_layer,
            )

    def get_conv_layers(self):
        if self.has_conv_layer:
            for layer in self.children():
                if isinstance(layer, ConvLayers):
                    return layer
        else:
            return -1

    def get_input_dimensions(self, flatten=True):
        for child in self.children():
            if hasattr(child, "get_input_dimensions"):
                return child.get_input_dimensions(flatten)

    @property
    def activation(self):
        return list(self.children())[-1].activation

    @property
    def last_activation(self):
        return list(self.children())[-1].last_activation


# Autoencoder base class
class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Sequential,
        decoder_mu=None,
        decoder_sig=None,
        homoscedestic_mode="none",
        use_cuda=False,
        decoder_cluster=None,
        skip=False,
    ):
        super(Autoencoder, self).__init__()
        self.skip = skip
        self.encoder = encoder

        if decoder_mu is None:
            self.decoder_mu = infer_decoder(encoder)
        else:
            self.decoder_mu = decoder_mu

        # decoder sig status
        if decoder_sig is None:
            self.decoder_sig_enabled = False
            self.decoder_full_cov_enabled = False
        else:
            self.decoder_sig_enabled = True
            self.decoder_sig = decoder_sig
            # check for full cov. decoder sig or regular diag cov.
            # based on the layer type, for now, only CholLayer can be
            # recognised for enabling full covariance calculation
            if isinstance(self.decoder_sig, CholLayer):
                self.decoder_full_cov_enabled = True
            else:
                self.decoder_full_cov_enabled = False

        # decoder cluster status
        if decoder_cluster is None:
            self.decoder_cluster_enabled = False
        else:
            self.decoder_cluster_enabled = True
            self.decoder_cluster = decoder_cluster

        # log noise mode
        self.homoscedestic_mode = homoscedestic_mode
        self.set_log_noise(homoscedestic_mode=self.homoscedestic_mode)

        # set cuda
        self.set_cuda(use_cuda)

    @property
    def latent_dim(self):
        return self.encoder.latent_dim

    def set_child_cuda(self, child, use_cuda=False):
        if isinstance(child, torch.nn.Sequential):
            for child in child.children():
                child.use_cuda = use_cuda
        else:
            child.use_cuda = use_cuda

    def set_cuda(self, use_cuda=False):

        self.set_child_cuda(self.encoder, use_cuda)
        self.set_child_cuda(self.decoder_mu, use_cuda)

        if self.decoder_sig_enabled:
            self.set_child_cuda(self.decoder_sig, use_cuda)
        if self.decoder_cluster_enabled:
            self.set_child_cuda(self.decoder_cluster, use_cuda)

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

        if homoscedestic_mode == "single":
            self.log_noise = Parameter(torch.FloatTensor([np.log(init_log_noise)] * 1))
        elif homoscedestic_mode == "every":
            if isinstance(self.encoder, Encoder):
                log_noise_size = self.encoder.get_input_dimensions()
            else:
                for child in self.encoder.children():
                    if hasattr(child, "get_input_dimensions"):
                        log_noise_size = child.get_input_dimensions()
                        break
            self.log_noise = Parameter(
                torch.FloatTensor([np.log(init_log_noise)] * log_noise_size)
            )
        else:
            self.log_noise = Parameter(torch.FloatTensor([[0.0]]), requires_grad=False)

    def reset_parameters(self):
        self._reset_nested_parameters(self.encoder)
        self._reset_nested_parameters(self.decoder_mu)
        if self.decoder_sig_enabled:
            self._reset_nested_parameters(self.decoder_sig)
        if self.decoder_cluster_enabled:
            self._reset_nested_parameters(self.decoder_cluster)

        return self

    def _reset_parameters(self, child_layer):
        if hasattr(child_layer, "reset_parameters"):
            child_layer.reset_parameters()

    def _reset_nested_parameters(self, network):
        if hasattr(network, "children"):
            for child_1 in network.children():
                for child_2 in child_1.children():
                    self._reset_parameters(child_2)
                    for child_3 in child_2.children():
                        self._reset_parameters(child_3)
                        for child_4 in child_3.children():
                            self._reset_parameters(child_4)
        return network

    def forward(self, x):
        if not self.skip:
            encoded = self.encoder(x)
            decoded_mu = self.decoder_mu(encoded)
            decoded_list = [decoded_mu]

            if self.decoder_sig_enabled:
                decoded_sig = self.decoder_sig(encoded)
                decoded_list.append(decoded_sig)

            if self.decoder_cluster_enabled:
                decoded_cluster = self.decoder_cluster(encoded)
                decoded_list.append(decoded_cluster)
        else:
            decoded_list = self.forward_skip(x)
        return tuple(decoded_list)

    def forward_skip(self, x):
        skip_outp = []
        max_layers_n = 0
        # count number of blocks
        for enc_layers in self.encoder:
            if isinstance(enc_layers, DenseLayers) or isinstance(
                enc_layers, ConvLayers
            ):
                max_layers_n += len(enc_layers.layers)

        max_layers_n -= 1

        # actually forward them now
        block_i = 0
        for enc_layers in self.encoder:
            if isinstance(enc_layers, DenseLayers) or isinstance(
                enc_layers, ConvLayers
            ):
                for block in enc_layers.layers:
                    x = block(x)
                    if block_i < max_layers_n:
                        skip_outp.append(x)
                        block_i += 1

            else:
                x = enc_layers(x)

        # encoded = x

        skip_outp.reverse()

        # check whether reshape is needed
        if self.encoder.has_dense_layer and self.encoder.has_conv_layer:
            need_reshape = True
        else:
            need_reshape = False

        block_i = 0
        for dec_layers in self.decoder_mu:
            if isinstance(dec_layers, DenseLayers) or isinstance(
                dec_layers, ConvLayers
            ):
                #
                # print("---X-SHAPE--")
                for l_i, block in enumerate(dec_layers.layers):
                    x = block(x)

                    if block_i < max_layers_n:
                        if (
                            need_reshape
                            and isinstance(dec_layers, DenseLayers)
                            and l_i == (len(dec_layers.layers) - 1)
                        ):
                            continue
                        x += skip_outp[block_i]
                        block_i += 1

                        # last layer
                        if block_i == max_layers_n and self.decoder_sig_enabled:
                            y_sig = self.decoder_sig(x)

            else:
                x = dec_layers(x)
                if need_reshape:
                    x += skip_outp[block_i]
                    block_i += 1

        decoded_list = [x]

        # handle decoder sig
        if self.decoder_sig_enabled:
            decoded_list.append(y_sig)

        # skip_outp = []
        # max_layers_n = 0
        # # count number of blocks
        # for enc_layers in self.encoder:
        #     if isinstance(enc_layers, DenseLayers) or isinstance(
        #         enc_layers, ConvLayers
        #     ):
        #         max_layers_n += len(enc_layers.layers)
        #
        # max_layers_n -= 1
        #
        # # actually forward them now
        # block_i = 0
        # for enc_layers in self.encoder:
        #     if isinstance(enc_layers, DenseLayers) or isinstance(
        #         enc_layers, ConvLayers
        #     ):
        #         for block in enc_layers.layers:
        #             x = block(x)
        #             if block_i < max_layers_n:
        #                 skip_outp.append(x)
        #                 block_i += 1
        #
        #     else:
        #         x = enc_layers(x)
        #
        # encoded = x
        #
        # skip_outp.reverse()
        #
        # # check whether reshape is needed
        # if self.encoder.has_dense_layer and self.encoder.has_conv_layer:
        #     need_reshape = True
        # else:
        #     need_reshape = False
        #
        # block_i = 0
        # for dec_layers in self.decoder_mu:
        #     if isinstance(dec_layers, DenseLayers) or isinstance(
        #         dec_layers, ConvLayers
        #     ):
        #         #
        #         # print("---X-SHAPE--")
        #         for l_i, block in enumerate(dec_layers.layers):
        #             x = block(x)
        #
        #             if block_i < max_layers_n:
        #                 if (
        #                     need_reshape
        #                     and isinstance(dec_layers, DenseLayers)
        #                     and l_i == (len(dec_layers.layers) - 1)
        #                 ):
        #                     continue
        #
        #                 x += skip_outp[block_i]
        #                 block_i += 1
        #
        #     else:
        #         x = dec_layers(x)
        #         if need_reshape:
        #             x += skip_outp[block_i]
        #             block_i += 1
        # decoded_list = [x]
        #
        # # handle decoder sig
        # if self.decoder_sig_enabled:
        #     block_i = 0
        #     for dec_layers in self.decoder_mu:
        #         if isinstance(dec_layers, DenseLayers) or isinstance(
        #             dec_layers, ConvLayers
        #         ):
        #             #
        #             # print("---X-SHAPE--")
        #             for l_i, block in enumerate(dec_layers.layers):
        #                 encoded = block(encoded)
        #
        #                 if block_i < max_layers_n:
        #                     if (
        #                         need_reshape
        #                         and isinstance(dec_layers, DenseLayers)
        #                         and l_i == (len(dec_layers.layers) - 1)
        #                     ):
        #                         continue
        #
        #                     encoded += skip_outp[block_i]
        #                     block_i += 1
        #
        #         else:
        #             encoded = dec_layers(encoded)
        #             if need_reshape:
        #                 encoded += skip_outp[block_i]
        #                 block_i += 1
        #     decoded_list.append(encoded)

        return decoded_list


###MODEL MANAGER: FIT/PREDICT
# BAE Base class
class BAE_BaseClass:
    def __init__(
        self,
        autoencoder: Autoencoder,
        num_samples=100,
        anchored=False,
        weight_decay=0.01,
        num_epochs=10,
        verbose=True,
        use_cuda=False,
        task="regression",
        learning_rate=0.01,
        learning_rate_sig=None,
        homoscedestic_mode="none",
        model_type="stochastic",
        model_name="BAE",
        scheduler_enabled=False,
        likelihood="gaussian",
        denoising_factor=0,
        cluster_weight=1.0,
        output_clamp=(0, 0),
        sparse=False,
    ):

        if autoencoder is None:
            raise ValueError(
                "Autoencoder cannot be None in instantiating BAE. Please pass an Autoencoder model."
            )

        # save kwargs
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
        self.cluster_weight = cluster_weight
        self.sparse = sparse

        # set output clamp
        if output_clamp == (0, 0):
            self.output_clamp = False
        else:
            self.output_clamp = output_clamp

        if learning_rate_sig is None:
            self.learning_rate_sig = learning_rate
        else:
            self.learning_rate_sig = learning_rate_sig

        # override homoscedestic mode of autoencoder
        if homoscedestic_mode is not None:
            autoencoder.set_log_noise(homoscedestic_mode=homoscedestic_mode)

        # revert to default
        else:
            homoscedestic_mode = autoencoder.homoscedestic_mode
        self.homoscedestic_mode = homoscedestic_mode

        # init BAE
        self.decoder_sigma_enabled = autoencoder.decoder_sig_enabled
        self.decoder_cluster_enabled = autoencoder.decoder_cluster_enabled
        self.decoder_full_cov_enabled = autoencoder.decoder_full_cov_enabled
        self.autoencoder = self.init_autoencoder(autoencoder)
        # self.set_optimisers(self.autoencoder, self.mode)
        self.optimisers = []
        # overwrite likelihood to be full gaussian, if decoder full cov is enabled
        if self.decoder_full_cov_enabled:
            self.likelihood = "full_gaussian"

        # init index of autoencoder outputs, due to possibility of multiple decoders
        if self.decoder_sigma_enabled and self.decoder_cluster_enabled:
            self.decoder_sig_index = 1
            self.decoder_cluster_index = 2
        elif self.decoder_sigma_enabled:
            self.decoder_sig_index = 1
        elif self.decoder_cluster_enabled:
            self.decoder_cluster_index = 1

        if self.decoder_cluster_enabled:
            self.kl_div_func = torch.nn.KLDivLoss()

    @property
    def latent_dim(self):
        if self.model_type == "list":
            return self.autoencoder[0].latent_dim
        elif self.model_type == "stochastic":
            return self.autoencoder.latent_dim

    def save_model_state(self, filename=None, folder_path="torch_model/"):
        create_dir(folder_path)
        if filename is None:
            temp = True
        if self.model_type == "list":
            for model_i, autoencoder in enumerate(self.autoencoder):
                if temp:
                    torch_filename = self.model_name + "_" + str(model_i) + ".pt"
                    torch_filename = "temp_" + torch_filename
                else:
                    torch_filename = temp
                torch.save(autoencoder.state_dict(), folder_path + torch_filename)

        else:  # stochastic model
            if temp:
                torch_filename = self.model_name + ".pt"
                torch_filename = "temp_" + torch_filename
            else:
                torch_filename = temp
            torch.save(self.autoencoder.state_dict(), folder_path + torch_filename)

    def load_model_state(self, filename=None, folder_path="torch_model/"):
        create_dir(folder_path)
        if filename is None:
            temp = True
        if self.model_type == "list":
            for model_i, autoencoder in enumerate(self.autoencoder):
                if temp:
                    torch_filename = self.model_name + "_" + str(model_i) + ".pt"
                    torch_filename = "temp_" + torch_filename
                else:
                    torch_filename = temp
                self.autoencoder[model_i].load_state_dict(
                    torch.load(folder_path + torch_filename)
                )
        else:  # stochastic model
            if temp:
                torch_filename = self.model_name + ".pt"
                torch_filename = "temp_" + torch_filename
            else:
                torch_filename = temp
            self.autoencoder.load_state_dict(torch.load(folder_path + torch_filename))

    def _get_homoscedestic_noise(self, model_type="list"):
        """
        Internal method to access the autoenocder's free log noise parameters, depending on the type of model.
        For example, this depends on whether the model is a list or a stochastic model
        Developers should override this to provide custom function to access the autoencoder's parameters

        This is similar in concept to the method `_predict_samples`
        """
        if model_type == "list":
            log_noise = np.array(
                [model.log_noise.detach().cpu().numpy() for model in self.autoencoder]
            )
            log_noise_mean = np.exp(log_noise).mean(0)
            log_noise_var = np.exp(log_noise).var(0)
            return (log_noise_mean, log_noise_var)
        elif model_type == "stochastic":
            log_noise = np.exp(self.autoencoder.log_noise.detach().cpu().numpy())
            return (log_noise, np.zeros_like(log_noise))

    def get_homoscedestic_noise(self, return_mean=True):
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

    def get_optimisers_list(
        self, autoencoder: Autoencoder, mode="mu", sigma_train="separate"
    ):
        optimiser_list = []
        if mode == "sigma":
            if autoencoder.decoder_sig_enabled:
                optimiser_list.append(
                    {
                        "params": autoencoder.decoder_sig.parameters(),
                        "lr": self.learning_rate_sig,
                    }
                )
                if sigma_train == "joint":  # for joint training
                    optimiser_list.append({"params": autoencoder.encoder.parameters()})
                    optimiser_list.append(
                        {"params": autoencoder.decoder_mu.parameters()}
                    )
                    optimiser_list.append({"params": autoencoder.log_noise})
        else:
            optimiser_list.append({"params": autoencoder.encoder.parameters()})
            optimiser_list.append({"params": autoencoder.decoder_mu.parameters()})
            optimiser_list.append({"params": autoencoder.log_noise})

        if self.decoder_cluster_enabled:
            optimiser_list.append({"params": autoencoder.decoder_cluster.parameters()})

        return optimiser_list

    def reset_parameters(self):
        if self.model_type == "list":
            for autoencoder in self.autoencoder:
                autoencoder.reset_parameters()
        else:
            self.autoencoder.reset_parameters()

    def get_optimisers(
        self, autoencoder: Autoencoder, mode="mu", sigma_train="separate"
    ):
        optimiser_list = self.get_optimisers_list(
            autoencoder, mode=mode, sigma_train=sigma_train
        )
        return torch.optim.Adam(optimiser_list, lr=self.learning_rate)
        # return torch.optim.SGD(optimiser_list, lr=self.learning_rate)

    def set_optimisers(
        self, autoencoder: Autoencoder, mode="mu", sigma_train="separate"
    ):
        if self.model_type == "list":
            self.optimisers = [
                self.get_optimisers(model, mode=mode, sigma_train=sigma_train)
                for model in self.autoencoder
            ]
        else:
            self.optimisers = self.get_optimisers(
                autoencoder, mode=mode, sigma_train=sigma_train
            )

        self.save_optimisers_state()

        return self.optimisers

    def save_optimisers_state(self):
        self.saved_optimisers_state = [
            optimiser.state_dict() for optimiser in self.optimisers
        ]

    def load_optimisers_state(self):
        for optimiser, state in zip(self.optimisers, self.saved_optimisers_state):
            optimiser.load_state_dict(state)

    def init_scheduler(self, half_iterations=None, min_lr=None, max_lr=None):
        # resort to internal stored values for scheduler
        if half_iterations is None or min_lr is None or max_lr is None:
            half_iterations = self.half_iterations
            min_lr = self.min_lr
            max_lr = self.max_lr
        else:
            self.half_iterations = half_iterations
            self.min_lr = min_lr
            self.max_lr = max_lr

        # handle model type to access optimiser hierarchy
        if self.model_type == "list":
            self.scheduler = [
                torch.optim.lr_scheduler.CyclicLR(
                    optimiser,
                    step_size_up=half_iterations,
                    base_lr=min_lr,
                    max_lr=max_lr,
                    cycle_momentum=False,
                )
                for optimiser in self.optimisers
            ]

        else:
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimisers,
                step_size_up=half_iterations,
                base_lr=min_lr,
                max_lr=max_lr,
                cycle_momentum=False,
            )
        self.scheduler_enabled = True

        # set init learning rate
        self.learning_rate = self.min_lr
        self.learning_rate_sig = self.min_lr
        return self.scheduler

    def set_learning_rate(self, learning_rate=None):
        # update internal learning rate
        if learning_rate is None:
            learning_rate = self.learning_rate
        else:
            self.learning_rate = learning_rate

        # handle access to optimiser parameters
        if self.model_type == "list":
            for optimiser in self.optimisers:
                for group in optimiser.param_groups:
                    group["lr"] = learning_rate

        if self.model_type == "stochastic":
            for group in self.optimisers.param_groups:
                group["lr"] = learning_rate

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

    def fit_one(self, x, y=None, mode="mu", inverse=False):
        """
        Template for vanilla fitting, developers are very likely to override this to provide custom fitting functions.
        """
        # extract input and output size from data
        # and convert into tensor, if not already

        # if denoising is enabled
        if self.denoising_factor:
            y = copy.deepcopy(x)
            x = x + self.denoising_factor * torch.randn(*x.shape)
            x = torch.clamp(x, 0.0, 1.0)

        if y is None:
            y = x
        x, y = self.convert_tensor(x, y)

        # train for n epochs
        loss = self.criterion(autoencoder=self.autoencoder, x=x, y=y, mode=mode)

        if inverse:
            loss *= -1

        # backpropagate
        self.zero_optimisers()
        loss.backward()
        self.step_optimisers()

        # if scheduler is enabled, update it
        if self.scheduler_enabled:
            self.step_scheduler()

        if self.use_cuda:
            x.cpu()
            y.cpu()
        return loss.item()

    def semisupervised_fit_one(self, x_inliers, x_outliers, mode="mu"):
        """
        Template for vanilla fitting, developers are very likely to override this to provide custom fitting functions.
        """
        # extract input and output size from data
        # and convert into tensor, if not already

        x_inliers = self.convert_tensor(x_inliers)
        x_outliers = self.convert_tensor(x_outliers)

        # train for n epochs
        loss = self.criterion(
            autoencoder=self.autoencoder, x=x_inliers, y=x_inliers, mode=mode
        )
        outlier_loss = -self.criterion(
            autoencoder=self.autoencoder, x=x_outliers, y=x_outliers, mode=mode
        )

        total_loss = loss + 0.01 * outlier_loss

        # backpropagate
        self.zero_optimisers()
        total_loss.backward()
        self.step_optimisers()

        # if scheduler is enabled, update it
        if self.scheduler_enabled:
            self.step_scheduler()

        if self.use_cuda:
            x_inliers.cpu()
            x_outliers.cpu()
        return loss.item()

    def normalised_fit(
        self,
        x_inliers,
        num_epochs=None,
        mode="mu",
        sigma_train="separate",
    ):
        """
        Overarching fitting function, to handle pytorch train loader or tensors
        """

        if self.scheduler_enabled:
            self.init_scheduler()

        if num_epochs is None:
            num_epochs = self.num_epochs

        # handle train loader
        if isinstance(x_inliers, torch.utils.data.dataloader.DataLoader):
            # save the number of iterations for KL weighting later
            self.num_iterations = len(x_inliers)

            for epoch in tqdm(range(num_epochs)):
                temp_loss = []
                for batch_idx, (data, target) in enumerate(x_inliers):
                    loss = self.normalised_fit_one(x_inliers=data, mode=mode)
                    temp_loss.append(loss)
                    self.losses.append(np.mean(temp_loss))
                self.print_loss(epoch, self.losses[-1])

        else:
            x_inliers = self.convert_tensor(x_inliers)

            for epoch in tqdm(range(num_epochs)):
                loss = self.normalised_fit_one(x_inliers=x_inliers, mode=mode)
                self.losses.append(loss)
                self.print_loss(epoch, self.losses[-1])
        return self

    def normalised_fit_one(self, x_inliers, mode="mu"):
        """
        Template for vanilla fitting, developers are very likely to override this to provide custom fitting functions.
        """
        # extract input and output size from data
        # and convert into tensor, if not already

        x_inliers = self.convert_tensor(x_inliers)
        # x_inliers = flatten_torch(x_inliers)

        # train for n epochs
        nll = torch.stack(
            [self.nll(ae, x_inliers, x_inliers, mode=mode) for ae in self.autoencoder]
        )

        loss = nll.mean()

        # normalised_loss = torch.log(torch.exp(-nll).mean(-1)).mean(-1).mean(-1)

        normalised_loss = torch.log(torch.exp(-nll.mean(-1)).mean(-1)).mean(0)

        # normalised_loss = torch.log(torch.exp(-nll.mean(-1).mean(-1))).mean(0)
        # normalised_loss = torch.log(torch.exp(-nll).mean(-1).mean(-1)).mean(0)

        # def criterion(self, autoencoder: Autoencoder, x, y=None, mode="sigma"):
        #     # likelihood
        #     nll = self.nll(autoencoder, x, y, mode)
        #     nll = nll.mean()
        #
        #     # prior loss
        #     prior_loss = self.log_prior_loss(model=autoencoder)
        #     prior_loss = prior_loss.mean()
        #     prior_loss *= self.weight_decay
        #
        #     return nll + prior_loss

        # normalised_loss = torch.stack(
        #     [(ae(x_inliers)[0] - x_inliers) ** 2 for ae in self.autoencoder]
        # )
        #
        # normalised_loss = torch.log(torch.exp(-normalised_loss).mean(-1)).mean()

        # for ae in self.autoencoder:
        #     loss_ = (ae(x_inliers)[0] - x_inliers) ** 2
        #     print(loss_)
        #

        # total_loss = loss - normalised_loss

        prior_loss = torch.stack(
            [self.log_prior_loss(model=ae) for ae in self.autoencoder]
        )
        prior_loss = prior_loss.mean()
        prior_loss *= self.weight_decay

        total_loss = loss + normalised_loss + prior_loss

        # total_loss = loss + normalised_loss
        # total_loss = loss
        # total_loss = normalised_loss

        # total_loss = loss

        # total_loss = loss
        # print("TOTAL LOSS:" + str(loss.item()))
        # print("NORMALISED LOSS:" + str(normalised_loss.item()))

        # nll_inliers_train = bae_ensemble.predict_samples(
        #     x_inliers_train, select_keys=["se"]
        # )
        #
        # nll_inliers_train_mean = nll_inliers_train[0][0].mean(-1)
        #
        # unnormalised_likelihood = np.exp(-nll_inliers_train_mean)
        # normaliser = unnormalised_likelihood.sum()
        # norm_ll = unnormalised_likelihood / normaliser
        #
        # log_normalizer = +np.log(np.exp(-nll_inliers_train_mean).sum())
        #

        # outlier_loss = -self.criterion(
        #     autoencoder=self.autoencoder, x=x_outliers, y=x_outliers, mode=mode
        # )
        #
        # total_loss = loss + 0.01 * outlier_loss

        # backpropagate
        self.zero_optimisers()
        total_loss.backward()
        self.step_optimisers()

        # if scheduler is enabled, update it
        if self.scheduler_enabled:
            self.step_scheduler()

        if self.use_cuda:
            x_inliers.cpu()
        return loss.item()

    def print_loss(self, epoch, loss):
        if self.verbose:
            print("LOSS #{}:{}".format(epoch, loss))

    def fit(
        self,
        x,
        y=None,
        mode="mu",
        num_epochs=None,
        sigma_train="separate",
        supervised=False,
        inverse=False,
    ):
        """
        Overarching fitting function, to handle pytorch train loader or tensors
        """

        if len(self.optimisers) == 0:
            self.set_optimisers(self.autoencoder, mode=mode, sigma_train=sigma_train)
        else:
            self.load_optimisers_state()

        if self.scheduler_enabled:
            self.init_scheduler()

        if num_epochs is None:
            num_epochs = self.num_epochs

        # handle train loader
        if isinstance(x, torch.utils.data.dataloader.DataLoader):
            # save the number of iterations for KL weighting later
            self.num_iterations = len(x)

            for epoch in tqdm(range(num_epochs)):
                temp_loss = []
                for batch_idx, (data, target) in enumerate(x):
                    if len(data) <= 2:
                        continue
                    if supervised:
                        loss = self.fit_one(x=data, y=target, mode=mode)
                    else:
                        loss = self.fit_one(x=data, y=data, mode=mode)
                    temp_loss.append(loss)
                    self.losses.append(np.mean(temp_loss))
                self.print_loss(epoch, self.losses[-1])

        else:
            if y is None:
                y = x
            x, y = self.convert_tensor(x, y)

            for epoch in tqdm(range(num_epochs)):
                loss = self.fit_one(x=x, y=y, mode=mode, inverse=inverse)
                self.losses.append(loss)
                self.print_loss(epoch, self.losses[-1])
        return self

    def semisupervised_fit(
        self,
        x_inliers,
        x_outliers,
        num_epochs=None,
        mode="mu",
        sigma_train="separate",
    ):
        """
        Overarching fitting function, to handle pytorch train loader or tensors
        """
        self.set_optimisers(self.autoencoder, mode=mode, sigma_train=sigma_train)

        if self.scheduler_enabled:
            self.init_scheduler()

        if num_epochs is None:
            num_epochs = self.num_epochs

        # handle train loader
        if isinstance(x_inliers, torch.utils.data.dataloader.DataLoader):
            # save the number of iterations for KL weighting later
            self.num_iterations = len(x_inliers)

            for epoch in tqdm(range(num_epochs)):
                temp_loss = []
                for batch_idx, (data, target) in enumerate(x_inliers):
                    loss = self.semisupervised_fit_one(
                        x_inliers=data, x_outliers=x_outliers, mode=mode
                    )
                    temp_loss.append(loss)
                    self.losses.append(np.mean(temp_loss))
                self.print_loss(epoch, self.losses[-1])

        else:
            x_inliers = self.convert_tensor(x_inliers)
            x_outliers = self.convert_tensor(x_outliers)

            for epoch in tqdm(range(num_epochs)):
                loss = self.semisupervised_fit_one(
                    x_inliers=x_inliers, x_outliers=x_outliers, mode=mode
                )
                self.losses.append(loss)
                self.print_loss(epoch, self.losses[-1])
        return self

    def nll(self, autoencoder: Autoencoder, x, y=None, mode="mu"):
        """
        Computes the NLL with the autoencoder, with the given likelihood.
        This depends on the given `mode` of whether it is hetero- or homoscedestic
        And also whether a decoder_sigma is enabled in the autoencoder.
        For now, mode="sigma" is only implemented for Gaussian likelihood and will only make sense in that setting.

        if y is provided,
        autoencoder : Autoencoder
            Autoencoder to compute the NLL with
        x : torch.Tensor
            input data
        y : torch.Tensor
            target data
        mode : str of "sigma" or "mu"
            If "sigma", the output of the autoencoder's decoder_mu and decoder_sigma will be used for the likelihood.
            If "mu", the autoencoder's decoder_mu and the homoscedestic term `log_noise` will be used for the likelihood.
        """
        # likelihood
        if y is None:
            y = x

        # get outputs of autoencoder, depending on the number of decoders
        # that is, if decoder sigma is enabled or not
        ae_output = autoencoder(x)
        y_pred_mu = ae_output[0]
        if self.decoder_sigma_enabled:
            y_pred_sig = ae_output[self.decoder_sig_index]

        # depending on the mode, we compute the nll
        if mode == "sigma":
            nll = self._nll(
                flatten_torch(y_pred_mu), flatten_torch(y), flatten_torch(y_pred_sig)
            )
        elif mode == "mu":
            # temporary switch for handling full gaussian mode
            if self.likelihood == "full_gaussian":
                self.likelihood = "gaussian"
                nll = self._nll(
                    flatten_torch(y_pred_mu), flatten_torch(y), autoencoder.log_noise
                )
                self.likelihood = "full_gaussian"
            else:
                nll = self._nll(
                    flatten_torch(y_pred_mu), flatten_torch(y), autoencoder.log_noise
                )

        if self.decoder_cluster_enabled:
            y_pred_cluster = ae_output[self.decoder_cluster_index]
            cluster_loss = self.cluster_loss(y_pred_cluster)
            return nll + self.cluster_weight * cluster_loss
        else:
            return nll

    def _nll(self, y_pred_mu, y, y_pred_sig=None):
        if self.likelihood == "gaussian":
            nll = -self.log_gaussian_loss_logsigma_torch(y_pred_mu, y, y_pred_sig)
        elif self.likelihood == "laplace":
            nll = torch.abs(y_pred_mu - y)
        elif self.likelihood == "bernoulli":
            nll = F.binary_cross_entropy(y_pred_mu, y, reduction="none")
        elif self.likelihood == "cbernoulli":
            nll = self.log_cbernoulli_loss_torch(y_pred_mu, y)
        elif self.likelihood == "truncated_gaussian":
            nll = -self.log_truncated_loss_torch(y_pred_mu, y, y_pred_sig)
        elif self.likelihood == "full_gaussian":
            nll = self.loss_full_gaussian_torch(y - y_pred_mu, *y_pred_sig)
        return nll

    def criterion(self, autoencoder: Autoencoder, x, y=None, mode="sigma"):
        # likelihood
        nll = self.nll(autoencoder, x, y, mode)
        nll = nll.mean()

        # prior loss
        prior_loss = self.log_prior_loss(model=autoencoder, L=1 if self.sparse else 2)
        prior_loss = prior_loss.mean()
        prior_loss *= self.weight_decay

        return nll + prior_loss

    def _predict_samples(
        self,
        x,
        model_type=0,
        select_keys=["y_mu", "y_sigma", "se", "bce", "cbce", "nll_homo", "nll_sigma"],
    ):
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
        # revert to default
        if model_type == 0:
            model_type = self.model_type

        # handle different model types
        if model_type == "list":
            y_preds = [
                list(self.calc_output_single(model, x, select_keys=select_keys))
                for model in self.autoencoder
            ]
        elif model_type == "stochastic":
            y_preds = [
                list(
                    self.calc_output_single(
                        self.autoencoder, x, select_keys=select_keys
                    )
                )
                for i in range(self.num_samples)
            ]

        return y_preds

    def predict_dataloader(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        exclude_keys: list = [],
    ):
        """
        Accumulate results from each test batch, instead of calculating all at one go.
        """
        final_results = {}

        for batch_idx, (data, target) in tqdm(enumerate(dataloader)):
            # predict new batch of results
            next_batch_result = self._predict(data, exclude_keys)

            # populate for first time
            if batch_idx == 0:
                final_results.update(next_batch_result)
            # append for subsequent batches
            else:
                for key in final_results.keys():
                    final_results[key] = np.concatenate(
                        (final_results[key], next_batch_result[key]), axis=0
                    )
        return final_results

    def predict_samples(
        self,
        x,
        select_keys=["y_mu", "y_sigma", "se", "bce", "cbce", "nll_homo", "nll_sigma"],
    ):
        """
        Returns raw samples from each forward pass of the AE models
        """

        x = self.convert_tensor(x)
        original_shape = np.array(x.shape)
        y_preds = self._predict_samples(x, select_keys=select_keys)
        y_preds = np.array(y_preds)

        try:
            y_preds = y_preds.reshape(
                self.num_samples, len(select_keys), *list(original_shape)
            )
        except Exception as e:
            # y_preds = y_preds.reshape(self.num_samples,len(select_keys),original_shape[0])
            y_preds = y_preds

        return y_preds

    def predict_cluster(self, x, encode=True):
        x = self.convert_tensor(x)
        if self.model_type == "list":
            if encode:
                return np.array(
                    [
                        ae.decoder_cluster(ae.encoder(x)).detach().cpu().numpy()
                        for ae in self.autoencoder
                    ]
                )
            else:
                return np.array(
                    [
                        ae.decoder_cluster(x).detach().cpu().numpy()
                        for ae in self.autoencoder
                    ]
                )

    def predict(self, x, select_keys=[]):
        """
        Handles various data types including torch dataloader

        For actual outputs of predictions, see `_predict(x)`

        """
        if isinstance(x, torch.utils.data.dataloader.DataLoader):
            return self.predict_dataloader(x, select_keys)
        else:  # numpy array or tensor
            return self._predict(x, select_keys)

    def _predict(self, x, select_keys: list = []):
        """
        Computes mean and variances of the BAE model outputs

        The outputs can be more than necessary, depending on the context.
        Users will need to filter which ones to include and exclude via keys

        Returns
        -------
        results : dict
        """

        # get raw samples from each model forward pass
        raw_samples = self.predict_samples(x)

        # return also the input
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # calculate mean and variance of model predictions
        mean_samples = raw_samples.mean(0)
        var_samples = raw_samples.var(0)

        # compute statistics of model outputs
        (
            y_mu_mean,
            y_sigma_mean,
            se_mean,
            bce_mean,
            cbce_mean,
            nll_homo_mean,
            nll_sigma_mean,
        ) = mean_samples
        (
            y_mu_var,
            y_sigma_var,
            se_var,
            bce_var,
            cbce_var,
            nll_homo_var,
            nll_sigma_var,
        ) = var_samples

        # total uncertainty = epistemic+aleatoric
        if self.decoder_sigma_enabled:
            total_unc = y_sigma_mean + y_mu_var
        elif self.homoscedestic_mode != "none":
            total_unc = (
                y_sigma_mean + self.get_homoscedestic_noise(return_mean=False)[0]
            )
        else:
            total_unc = y_mu_var

        # calculate waic
        waic_se = se_mean + se_var
        waic_homo = nll_homo_mean + nll_homo_var
        waic_nll = nll_sigma_mean + nll_sigma_var

        # bce loss
        waic_bce = bce_mean + bce_var
        waic_cbce = cbce_mean + cbce_var

        # filter out unnecessary outputs based on model settings of computing aleatoric uncertainty
        # if self.decoder_sigma_enabled == False:
        #     exclude_keys = exclude_keys+["aleatoric","aleatoric_var", "nll_sigma_var", "nll_sigma_mean", "nll_sigma_waic"]
        # if self.homoscedestic_mode == "none":
        #     exclude_keys = exclude_keys+["nll_homo_mean", "nll_homo_var", "nll_homo_waic"]
        # if self.homoscedestic_mode == "none" and self.decoder_sigma_enabled == False:
        #     exclude_keys = exclude_keys+["total_unc"]

        return_dict = {
            "input": x,
            "mu": y_mu_mean,
            "epistemic": y_mu_var,
            "aleatoric": y_sigma_mean,
            "aleatoric_var": y_sigma_var,
            "total_unc": total_unc,
            "bce_mean": bce_mean,
            "bce_var": bce_var,
            "bce_waic": waic_bce,
            "cbce_mean": cbce_mean,
            "cbce_var": cbce_var,
            "cbce_waic": waic_cbce,
            "se_mean": se_mean,
            "se_var": se_var,
            "se_waic": waic_se,
            "nll_homo_mean": nll_homo_mean,
            "nll_homo_var": nll_homo_var,
            "nll_homo_waic": waic_homo,
            "nll_sigma_mean": nll_sigma_mean,
            "nll_sigma_var": nll_sigma_var,
            "nll_sigma_waic": waic_nll,
        }

        return_dict = {
            filtered_key: value
            for filtered_key, value in return_dict.items()
            if filtered_key in select_keys
        }
        return return_dict

    def bce_loss_np(self, y_pred, y_true):
        bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        bce = np.nan_to_num(bce, nan=0, posinf=100, neginf=-100)
        return bce

    def cbce_loss_np(self, y_pred, y_true):
        cbce = self.log_cbernoulli_loss_np(y_pred, y_true)
        cbce = np.nan_to_num(cbce, nan=0, posinf=100, neginf=-100)
        return cbce

    def _get_mu_sigma_single(self, autoencoder, x):
        ae_output = autoencoder(x)
        y_mu = ae_output[0]
        if self.decoder_sigma_enabled:
            y_sigma = ae_output[self.decoder_sig_index]
        else:
            y_sigma = torch.ones_like(y_mu)
        if self.decoder_cluster_enabled:
            y_cluster = ae_output[self.decoder_cluster_index].detach().cpu().numpy()

        # convert to numpy
        y_mu = y_mu.detach().cpu().numpy()

        if self.decoder_full_cov_enabled:
            # convert chol tril and log noise to numpy
            y_sigma = tuple([y_sigma_i.detach().cpu().numpy() for y_sigma_i in y_sigma])
        else:
            y_sigma = y_sigma.detach().cpu().numpy()
            y_sigma = flatten_np(y_sigma)
        log_noise = autoencoder.log_noise.detach().cpu().numpy()

        if self.decoder_cluster_enabled:
            return flatten_np(y_mu), y_sigma, log_noise, y_cluster
        else:
            return flatten_np(y_mu), y_sigma, log_noise

    def calc_output_single(
        self,
        autoencoder,
        x,
        select_keys=["y_mu", "y_sigma", "se", "bce", "cbce", "nll_homo", "nll_sigma"],
    ):
        """
        Computes the output of autoencoder per sample. Given the selected keys, we specify the output not only to be the reconstructed signal, but options of
        squared error (`se`), Gaussian negative loglikelihood (`nll_homo`), etc. This function is used later for every sampled autoencoder from the posterior weights.
        """
        # per sample
        if self.decoder_cluster_enabled:
            y_mu, y_sigma, log_noise, y_cluster = self._get_mu_sigma_single(
                autoencoder, x
            )
        else:
            y_mu, y_sigma, log_noise = self._get_mu_sigma_single(autoencoder, x)

        # clamp it to min max
        if self.output_clamp:
            y_mu = np.clip(y_mu, a_min=self.output_clamp[0], a_max=self.output_clamp[1])

        # flatten x into numpy array
        x = flatten_np(x.detach().cpu().numpy())

        return self._calc_output_single(
            x, y_mu=y_mu, y_sigma=y_sigma, log_noise=log_noise, select_keys=select_keys
        )

    def _calc_output_single(
        self,
        x,
        y_mu,
        y_sigma,
        log_noise,
        select_keys=["y_mu", "y_sigma", "se", "bce", "cbce", "nll_homo", "nll_sigma"],
    ):
        if not isinstance(y_sigma, np.ndarray):
            y_sigma = y_sigma.detach().cpu().numpy()
        if not isinstance(x, np.ndarray):
            x = x.detach().cpu().numpy()
        if not isinstance(y_mu, np.ndarray):
            y_mu = y_mu.detach().cpu().numpy()
        if not isinstance(y_mu, np.ndarray):
            log_noise = log_noise.detach().cpu().numpy()

        # return keys
        outputs = []

        for key in select_keys:
            if key == "y_mu":
                outputs.append(y_mu)
            elif key == "y_sigma":
                if self.decoder_full_cov_enabled:
                    exp_y_sig = np.exp(y_sigma[-1])
                else:
                    exp_y_sig = np.exp(y_sigma)
                outputs.append(exp_y_sig)
            elif key == "se":
                se = (y_mu - x) ** 2
                outputs.append(se)
            elif key == "bce":
                bce = self.bce_loss_np(y_mu, x)
                outputs.append(bce)
            elif key == "cbce":
                cbce = self.cbce_loss_np(y_mu, x)
                outputs.append(cbce)
            elif key == "nll_homo":
                if self.likelihood == "truncated_gaussian":
                    nll_homo = (
                        -self.log_truncated_loss_torch_v2(
                            torch.from_numpy(y_mu),
                            torch.from_numpy(x),
                            torch.from_numpy(log_noise),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    nll_homo = -self.log_gaussian_loss_logsigma_np(y_mu, x, log_noise)
                outputs.append(nll_homo)
            elif key == "nll_sigma":
                if self.decoder_full_cov_enabled:
                    nll_sigma = -self.loss_full_gaussian_np(x - y_mu, *y_sigma)
                else:
                    if isinstance(y_sigma, torch.Tensor):
                        y_sigma = y_sigma.detach().cpu().numpy()
                    if self.likelihood == "truncated_gaussian":
                        nll_sigma = (
                            -self.log_truncated_loss_torch_v2(
                                torch.from_numpy(y_mu),
                                torch.from_numpy(x),
                                torch.from_numpy(y_sigma),
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        nll_sigma = -self.log_gaussian_loss_logsigma_np(
                            y_mu, x, y_sigma
                        )
                outputs.append(nll_sigma)
        return tuple(outputs)

    def log_gaussian_loss_logsigma_torch(self, y_pred, y_true, log_sigma):
        log_likelihood = (-((y_true - y_pred) ** 2) * torch.exp(-log_sigma) * 0.5) - (
            0.5 * log_sigma
        )
        return log_likelihood

    def log_gaussian_loss_sigma_2_torch(self, y_pred, y_true, sigma_2):
        log_likelihood = (-((y_true - y_pred) ** 2) / (2 * sigma_2)) - (
            0.5 * torch.log(sigma_2)
        )
        return log_likelihood

    def log_gaussian_loss_logsigma_np(self, y_pred, y_true, log_sigma):
        log_likelihood = (-((y_true - y_pred) ** 2) * np.exp(-log_sigma) * 0.5) - (
            0.5 * log_sigma
        )
        return log_likelihood

    def log_gaussian_loss_sigma_2_np(self, y_pred, y_true, sigma_2):
        log_likelihood = (-((y_true - y_pred) ** 2) / (2 * sigma_2)) - (
            0.5 * np.log(sigma_2)
        )
        return log_likelihood

    def cluster_loss(self, cluster_output):
        # calculate target distribution
        tar_dist = cluster_output ** 2 / torch.sum(cluster_output, axis=0)
        tar_dist = torch.transpose(
            torch.transpose(tar_dist, 0, 1) / torch.sum(tar_dist, axis=1), 0, 1
        )

        # calculate KL divergence
        loss_clust = (
            -self.kl_div_func(torch.log(cluster_output), tar_dist)
            / cluster_output.shape[0]
        )
        return loss_clust

    def loss_full_gaussian_torch(self, y, chol_lower_tri, log_noise):
        # handle batch size of 1 by unsqueezing them
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
            chol_lower_tri = chol_lower_tri.unsqueeze(0)
            log_noise = log_noise.unsqueeze(0)

        # calculate reconstruction loss
        chol_y = torch.matmul(
            torch.transpose(torch.exp(chol_lower_tri), 2, 1), y.unsqueeze(-1)
        )
        chol_recon_loss = torch.matmul(torch.transpose(chol_y, 2, 1), chol_y)
        chol_recon_loss = chol_recon_loss.view(-1, 1)

        # calculate log determinant
        log_det = (-2 * (log_noise.sum(-1))).view(-1, 1)
        return chol_recon_loss + log_det

    def loss_full_gaussian_np(self, y, chol_lower_tri, log_noise):
        # handle batch size of 1 by unsqueezing them
        if len(y.shape) == 1:
            y = np.expand_dims(y, 0)
            chol_lower_tri = np.expand_dims(chol_lower_tri, 0)
            log_noise = np.expand_dims(log_noise, 0)

        # calculate reconstruction loss
        chol_y = np.matmul(
            np.transpose(np.exp(chol_lower_tri), (0, 2, 1)), np.expand_dims(y, -1)
        )
        chol_recon_loss = np.matmul(np.transpose(chol_y, (0, 2, 1)), chol_y)
        chol_recon_loss = chol_recon_loss.reshape(-1, 1)

        # calculate log determinant
        log_det = (-2 * (log_noise.sum(-1))).reshape(-1, 1)
        return chol_recon_loss + log_det

    def log_truncated_loss_torch(self, y_pred_mu, y_true, y_pred_sig, use_cuda=None):
        if hasattr(self, "trunc_g") == False:
            if use_cuda is None:
                use_cuda = self.use_cuda
            self.trunc_g = TruncatedGaussian(use_cuda=use_cuda)
        nll_trunc_g = self.trunc_g.truncated_log_pdf(
            y_true, y_pred_mu, torch.exp(y_pred_sig)
        )
        return nll_trunc_g

    def log_truncated_loss_torch_v2(self, y_pred_mu, y_true, y_pred_sig):
        self.trunc_g = TruncatedGaussian(use_cuda=False)
        nll_trunc_g = self.trunc_g.truncated_log_pdf(
            y_true, y_pred_mu, torch.exp(y_pred_sig)
        )
        return nll_trunc_g

    def log_cbernoulli_loss_torch(self, y_pred_mu, y_true):
        if hasattr(self, "cb") == False:
            self.cb = CB_Distribution()
        nll_cb = self.cb.log_cbernoulli_loss_torch(y_pred_mu, y_true)
        return nll_cb

    def log_cbernoulli_loss_np(self, y_pred_mu, y_true):
        if hasattr(self, "cb") == False:
            self.cb = CB_Distribution()
        return self.cb.log_cbernoulli_loss_np(
            torch.from_numpy(y_pred_mu), torch.from_numpy(y_true)
        )

    def log_prior_loss(self, model, mu=torch.Tensor([0.0]), L=2):
        # prior 0 ,1
        if self.anchored:
            mu = model.anchored_prior

        if self.use_cuda:
            mu = mu.cuda()

        weights = torch.cat([parameter.flatten() for parameter in model.parameters()])

        if L >= 2:
            prior_loss = torch.pow((weights - mu), L)
        else:
            prior_loss = torch.abs(weights - mu)
        return prior_loss

    def get_anchored_weight(self, model):
        model_weights = torch.cat(
            [parameter.flatten() for parameter in model.parameters()]
        )
        anchored_weights = torch.ones_like(model_weights) * model_weights.detach()
        return anchored_weights

    def init_anchored_weight(self, model):
        model.anchored_prior = self.get_anchored_weight(model)
        return model

    def convert_tensor(self, x, y=None):
        if "Tensor" not in type(x).__name__:
            x = Variable(torch.from_numpy(x).float())
        if y is not None:
            if "Tensor" not in type(y).__name__:
                if self.task == "classification":
                    y = Variable(torch.from_numpy(y).long())
                else:
                    y = Variable(torch.from_numpy(y).float())

        # use cuda
        if self.use_cuda:
            if y is None:
                return x.cuda()
            else:
                return x.cuda(), y.cuda()
        else:
            if y is None:
                return x
            else:
                return x, y

    def predict_latent_samples_(self, x):
        """
        Function for model to pass a forward to get the samples of latent values
        """
        # handle different model types
        if self.model_type == "list":
            latent_samples = torch.stack(
                [self._forward_latent_single(model, x) for model in self.autoencoder]
            )
        elif self.model_type == "stochastic":
            latent_samples = torch.stack(
                [
                    self._forward_latent_single(self.autoencoder, x)
                    for i in range(self.num_samples)
                ]
            )
        return latent_samples

    def predict_latent_samples(self, x):
        """
        Return latent samples (from bottle neck) of shape (M samples, n examples, latent_dim)
        """

        x = self.convert_tensor(x)
        latent_samples = self.predict_latent_samples_(x).detach().cpu().numpy()

        return latent_samples

    def _forward_latent_single(self, model, x):
        return model.encoder(x)

    def predict_latent(self, x, transform_pca=True):
        """
        Since BAE is probabilistic, we can obtain mean and variance of the latent dimensions for each data
        """
        latent_samples = self.predict_latent_samples(x)

        if transform_pca:
            pca = PCA(n_components=2)
            latent_data_ = np.array(
                [
                    pca.fit_transform(latent_samples[latent_sample_i])
                    for latent_sample_i in range(latent_samples.shape[0])
                ]
            )
            latent_samples = latent_data_
        return latent_samples.mean(0), latent_samples.var(0)

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
        return self

    def forward_samples(self, x):
        x = self.convert_tensor(x)
        x
