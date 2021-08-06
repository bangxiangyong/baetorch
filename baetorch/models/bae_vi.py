from ..models.base_autoencoder import (
    ConvLayers,
    DenseLayers,
    BAE_BaseClass,
    Autoencoder,
    flatten_torch,
    flatten_np,
    Encoder,
)
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
from torch.autograd import Variable
from sklearn.decomposition import PCA
import numpy as np

# VI
class VariationalLinear(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        prior_mu=0.0,
        prior_sigma_1=1.0,
        prior_sigma_2=0.1,
        prior_pi=0.5,
    ):
        super(VariationalLinear, self).__init__()
        self.weight_mu = Parameter(torch.Tensor(output_size, input_size))
        self.weight_sigma = Parameter(torch.Tensor(output_size, input_size))

        self.bias_mu = Parameter(torch.Tensor(output_size))
        self.bias_sigma = Parameter(torch.Tensor(output_size))

        self.prior_mu = Parameter(torch.FloatTensor([prior_mu]), requires_grad=False)
        self.prior_sigma = Parameter(
            torch.FloatTensor([prior_sigma_1]), requires_grad=False
        )
        self.prior_mu_ = prior_mu
        self.prior_sigma_ = prior_sigma_1

        # scale gaussian mixture
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi

        self.input_size = input_size
        self.output_size = output_size
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "weight_mu"):
            self.weight_mu.data.normal_(self.prior_mu_, self.prior_sigma_ * 0.1)
            self.weight_sigma.data = torch.ones_like(self.weight_sigma) * -3
            self.bias_mu.data.normal_(self.prior_mu_, self.prior_sigma_ * 0.1)
            self.bias_sigma.data = torch.ones_like(self.bias_sigma) * -3

    def log_gaussian_loss_sigma_2_torch(self, y_pred, y_true, sigma_2):
        log_likelihood = (-((y_true - y_pred) ** 2) / (2 * sigma_2)) - (
            0.5 * torch.log(sigma_2)
        )
        return log_likelihood

    def gaussian_loss_sigma_torch(self, y_pred, y_true, sigma):
        likelihood = (
            (1 / (sigma * 2.506))
            * torch.exp(-((y_true - y_pred) ** 2))
            / (2 * sigma ** 2)
        )
        return likelihood

    def kl_loss_prior_mixture(self, weight, weight_mu, weight_sigma):
        q_variational_log_prob = self.log_gaussian_loss_sigma_2_torch(
            weight, weight_mu, weight_sigma
        )
        prior_log_prob = torch.log(
            self.prior_pi
            * self.gaussian_loss_sigma_torch(weight, self.prior_mu, self.prior_sigma_1)
            + (1 - self.prior_pi)
            * self.gaussian_loss_sigma_torch(weight, self.prior_mu, self.prior_sigma_2)
        )
        kl_loss = (q_variational_log_prob - prior_log_prob).mean()

        return kl_loss

    def kl_loss_prior_gaussian(self, weight, weight_mu, weight_sigma):
        q_variational_log_prob = self.log_gaussian_loss_sigma_2_torch(
            weight, weight_mu, weight_sigma
        )
        prior_log_prob = self.log_gaussian_loss_sigma_2_torch(
            weight, self.prior_mu, self.prior_sigma
        )
        kl_loss = (q_variational_log_prob - prior_log_prob).mean()

        return kl_loss

    def weight_sample(self):
        weight = self.weight_mu + F.softplus(self.weight_sigma) * torch.randn_like(
            self.weight_mu
        )
        return weight

    def bias_sample(self):
        bias = self.bias_mu + F.softplus(self.bias_sigma) * torch.randn_like(
            self.bias_mu
        )
        return bias

    def forward(self, x):
        # draw samples for weight and bias
        weight = self.weight_sample()
        bias = self.bias_sample()
        kl_loss = self.kl_loss_prior_mixture(
            weight, self.weight_mu, F.softplus(self.weight_sigma)
        )
        y = F.linear(x, weight, bias)
        return y, kl_loss


class VariationalBaseConv:
    def __init__(
        self, prior_mu=0.0, prior_sigma_1=1.0, prior_sigma_2=0.1, prior_pi=0.5
    ):
        self.weight_mu = Parameter(torch.Tensor(*self.weight.shape))
        self.weight_sigma = Parameter(torch.Tensor(*self.weight.shape))

        self.bias_mu = Parameter(torch.Tensor(*self.bias.shape))
        self.bias_sigma = Parameter(torch.Tensor(*self.bias.shape))

        self.prior_mu = Parameter(torch.FloatTensor([prior_mu]), requires_grad=False)
        self.prior_sigma = Parameter(
            torch.FloatTensor([prior_sigma_1]), requires_grad=False
        )
        self.prior_mu_ = prior_mu
        self.prior_sigma_ = prior_sigma_1

        # scale gaussian mixture
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi

    def reset_parameters(self):
        if hasattr(self, "weight_mu"):
            self.weight_mu.data.normal_(self.prior_mu_, self.prior_sigma_ * 0.1)
            self.weight_sigma.data = torch.ones_like(self.weight_sigma) * -3
            self.bias_mu.data.normal_(self.prior_mu_, self.prior_sigma_ * 0.1)
            self.bias_sigma.data = torch.ones_like(self.bias_sigma) * -3

    def log_gaussian_loss_sigma_2_torch(self, y_pred, y_true, sigma_2):
        log_likelihood = (-((y_true - y_pred) ** 2) / (2 * sigma_2)) - (
            0.5 * torch.log(sigma_2)
        )
        return log_likelihood

    def gaussian_loss_sigma_torch(self, y_pred, y_true, sigma):
        likelihood = (
            (1 / (sigma * 2.506))
            * torch.exp(-((y_true - y_pred) ** 2))
            / (2 * sigma ** 2)
        )
        return likelihood

    def kl_loss_prior_mixture(self, weight, weight_mu, weight_sigma):
        q_variational_log_prob = self.log_gaussian_loss_sigma_2_torch(
            weight, weight_mu, weight_sigma
        )
        prior_log_prob = torch.log(
            self.prior_pi
            * self.gaussian_loss_sigma_torch(weight, self.prior_mu, self.prior_sigma_1)
            + (1 - self.prior_pi)
            * self.gaussian_loss_sigma_torch(weight, self.prior_mu, self.prior_sigma_2)
        )
        kl_loss = (q_variational_log_prob - prior_log_prob).mean()

        return kl_loss

    def kl_loss_prior_gaussian(self, weight, weight_mu, weight_sigma):
        q_variational_log_prob = self.log_gaussian_loss_sigma_2_torch(
            weight, weight_mu, weight_sigma
        )
        prior_log_prob = self.log_gaussian_loss_sigma_2_torch(
            weight, self.prior_mu, self.prior_sigma
        )
        kl_loss = (q_variational_log_prob - prior_log_prob).mean()

        return kl_loss

    def weight_sample(self):
        weight = self.weight_mu + F.softplus(self.weight_sigma) * torch.randn_like(
            self.weight_mu
        )
        return weight

    def bias_sample(self):
        bias = self.bias_mu + F.softplus(self.bias_sigma) * torch.randn_like(
            self.bias_mu
        )
        return bias


class VariationalConv2D(Conv2d, VariationalBaseConv):
    def __init__(self, **kwargs):
        Conv2d.__init__(self, **kwargs)
        VariationalBaseConv.__init__(self)
        self.reset_parameters()

    def reset_parameters(self):
        Conv2d.reset_parameters(self)
        VariationalBaseConv.reset_parameters(self)

    def forward(self, x):
        # draw samples for weight and bias
        weight = self.weight_sample()
        bias = self.bias_sample()
        kl_loss = self.kl_loss_prior_mixture(
            weight, self.weight_mu, F.softplus(self.weight_sigma)
        )
        y = F.conv2d(
            x, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        return y, kl_loss


class VariationalConv2DTranspose(ConvTranspose2d, VariationalBaseConv):
    def __init__(self, **kwargs):
        ConvTranspose2d.__init__(self, **kwargs)
        VariationalBaseConv.__init__(self)
        self.reset_parameters()

    def reset_parameters(self):
        ConvTranspose2d.reset_parameters(self)
        VariationalBaseConv.reset_parameters(self)

    def forward(self, x):
        # draw samples for weight and bias
        weight = self.weight_sample()
        bias = self.bias_sample()
        kl_loss = self.kl_loss_prior_mixture(
            weight, self.weight_mu, F.softplus(self.weight_sigma)
        )
        y = F.conv_transpose2d(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

        return y, kl_loss


class VariationalConv2DLayers(ConvLayers):
    def __init__(self, **kwargs):
        super(VariationalConv2DLayers, self).__init__(
            **kwargs, layer_type=[VariationalConv2D, VariationalConv2DTranspose]
        )
        # total_kl_loss = Variable(torch.tensor([0.]))

    def forward(self, x):
        if isinstance(x, tuple):
            x, total_kl_loss = x
            total_kl_loss = total_kl_loss[0]
        else:
            total_kl_loss = Variable(torch.tensor([0.0]))
            if self.use_cuda:
                total_kl_loss = total_kl_loss.cuda()
        # apply relu
        for layer_index, layer in enumerate(self.layers):
            # first sub layer is VI
            x, kl_loss = layer[0](x)
            total_kl_loss = total_kl_loss + kl_loss

            # relu, and other sub layers
            if len(layer) > 1:
                for sub_layer in layer[1:]:
                    x = sub_layer(x)
        return x, total_kl_loss


class VariationalDenseLayers(DenseLayers):
    def __init__(self, **kwargs):
        super(VariationalDenseLayers, self).__init__(
            **kwargs, layer_type=VariationalLinear
        )

        self.activation_ids = []
        for layer_id, layer in enumerate(self.layers):
            if isinstance(layer, VariationalLinear) == False:
                self.activation_ids.append(layer_id)
        self.last_layer_id = len(self.layers) - 1

    def forward(self, x):
        # if its a tuple, we expect it to carry the total_kl_loss from previous layer
        if isinstance(x, tuple):
            x, total_kl_loss = x
            total_kl_loss = total_kl_loss[0]
        else:
            total_kl_loss = Variable(torch.tensor([0.0]))
            if self.use_cuda:
                total_kl_loss = total_kl_loss.cuda()

        for layer_index, layer in enumerate(self.layers):
            # if activation layer, the return does not include kl_loss
            if layer_index in self.activation_ids:
                x = layer(x)
            # otherwise, the layer is variational layer, it returns kl_loss
            # we add the kl_loss to total
            else:
                x, kl_loss = layer(x)
                total_kl_loss = total_kl_loss + kl_loss

        return x, total_kl_loss


class BAE_VI(BAE_BaseClass):
    """
    Note: The output of autoencoder networks (encoder, decoder) consists of the prediction and KL-Loss.

    For an autoencoder with decoder_sigma, the nested tuples becomes ((decoder_mu, decoder_sigma), (output,kl_loss))
    """

    def __init__(self, *args, model_name="BAE_VI", num_train_samples=1, **kwargs):
        if num_train_samples <= 1:
            num_train_samples = 1
        self.num_train_samples = num_train_samples  # for training averaging
        super(BAE_VI, self).__init__(
            *args, model_name=model_name, model_type="stochastic", **kwargs
        )

    def nll_kl_loss(self, autoencoder, x, y=None, mode="sigma"):
        # likelihood
        if y is None:
            y = x

        # forward sample of autoencoder
        if self.decoder_sigma_enabled:
            decoded_mu, decoded_sigma = autoencoder(x)
            y_pred_mu, kl_mu = decoded_mu
            y_pred_sig, kl_sig = decoded_sigma
            kl_loss = kl_mu + kl_sig
        else:
            y_pred_mu, kl_loss = autoencoder(x)

        # get actual nll according to selected likelihood and mode
        if mode == "sigma":
            nll = self._nll(
                flatten_torch(y_pred_mu), flatten_torch(y), flatten_torch(y_pred_sig)
            )
        elif mode == "mu":
            nll = self._nll(
                flatten_torch(y_pred_mu), flatten_torch(y), autoencoder.log_noise
            )

        return nll, kl_loss

    def criterion(self, autoencoder, x, y=None, mode="sigma"):
        # likelihood + kl_loss
        for num_train_sample in range(self.num_train_samples):
            if num_train_sample == 0:
                nll, kl_loss = self.nll_kl_loss(autoencoder, x, y, mode)
            else:
                nll_temp, kl_loss_temp = self.nll_kl_loss(autoencoder, x, y, mode)
                nll = nll + nll_temp
                kl_loss = kl_loss + kl_loss_temp
        nll = nll / self.num_train_samples
        kl_loss = kl_loss / self.num_train_samples
        nll = nll.mean()

        # scale kl by a constant hyperparameter
        kl_loss *= self.weight_decay

        return nll + kl_loss

    def _forward_latent_single(self, model, x):
        return model.encoder(x)[0]

    def _get_mu_sigma_single(self, autoencoder, x):
        if self.decoder_sigma_enabled:
            decoded_mu, decoded_sigma = autoencoder(x)
            y_mu, kl_mu = decoded_mu
            y_sigma, kl_sig = decoded_sigma
            del kl_mu
            del kl_sig
        else:
            y_mu, kl_loss = autoencoder(x)
            del kl_loss
            y_sigma = torch.ones_like(y_mu)

        # convert to numpy
        y_mu = y_mu.detach().cpu().numpy()
        y_sigma = y_sigma.detach().cpu().numpy()
        log_noise = autoencoder.log_noise.detach().cpu().numpy()

        return flatten_np(y_mu), flatten_np(y_sigma), log_noise

    def convert_conv_vi(self, conv_layer):
        conv_params = {}
        for key, val in conv_layer.__dict__.items():
            exclude_params = [
                "activation_layer",
                "model_kwargs",
                "training",
                "conv2d_layer_type",
                "conv2d_trans_layer_type",
            ]
            if key[0] != "_" and key not in exclude_params:
                conv_params.update({key: val})
        conv_vi = VariationalConv2DLayers(**conv_params, reverse_params=False)
        return conv_vi

    def convert_dense_vi(self, dense_layer):
        dense_params = {}
        for key, val in dense_layer.__dict__.items():
            exclude_params = ["activation_layer", "model_kwargs"]
            if key[0] != "_" and key not in exclude_params:
                dense_params.update({key: val})
        converted_dense = VariationalDenseLayers(**dense_params)
        return converted_dense

    def convert_layer(self, layer):
        if isinstance(layer, ConvLayers):
            return self.convert_conv_vi(layer)
        if isinstance(layer, DenseLayers):
            return self.convert_dense_vi(layer)
        else:
            return layer

    def convert_torch_sequential(self, torch_sequential):
        converted_branch = []
        for layer in torch_sequential.children():
            converted_branch.append(self.convert_layer(layer))
        return torch.nn.Sequential(*converted_branch)

    def convert_autoencoder(self, autoencoder=Autoencoder):
        encoder = (
            self.convert_torch_sequential(autoencoder.encoder)
            if isinstance(autoencoder.encoder, torch.nn.Sequential)
            else self.convert_layer(autoencoder.encoder)
        )
        decoder_mu = (
            self.convert_torch_sequential(autoencoder.decoder_mu)
            if isinstance(autoencoder.decoder_mu, torch.nn.Sequential)
            else self.convert_layer(autoencoder.decoder_mu)
        )

        if autoencoder.decoder_sig_enabled:
            decoder_sig = (
                self.convert_torch_sequential(autoencoder.decoder_sig)
                if isinstance(autoencoder.decoder_sig, torch.nn.Sequential)
                else self.convert_layer(autoencoder.decoder_sig)
            )
        else:
            decoder_sig = None
        return Autoencoder(
            encoder=encoder, decoder_mu=decoder_mu, decoder_sig=decoder_sig
        )


class VAELinear(VariationalLinear):
    def forward(self, weight_mu, weight_sigma):
        # draw samples for weight and bias
        self.weight_mu.data = weight_mu
        self.weight_sigma.data = weight_sigma

        weight = self.weight_sample()

        kl_loss = self.kl_loss_prior_gaussian(
            weight, self.weight_mu, F.softplus(self.weight_sigma)
        )

        return weight, kl_loss


class VAE_Module(Autoencoder):
    def __init__(self, **kwargs):
        super(VAE_Module, self).__init__(**kwargs)
        (
            self.encoder,
            self.latent_layer_mu,
            self.latent_layer_sigma,
            self.latent_layer_vi,
        ) = self.infer_latent_layers(self.encoder)

        # set cuda
        self.set_cuda(self.use_cuda)

    def infer_latent_layers(self, encoder: Encoder):
        new_encoder = []

        for layer in encoder:
            if isinstance(layer, DenseLayers):
                encoder_dense_layer = layer

                if len(encoder_dense_layer.architecture) == 1:
                    latent_input_size = encoder_dense_layer.architecture[-1]
                    encoder_dense = torch.nn.Sequential(
                        torch.nn.Linear(
                            encoder_dense_layer.input_size, latent_input_size
                        ),
                        torch.nn.ReLU(),
                    )
                elif len(encoder_dense_layer.architecture) == 0:
                    latent_input_size = encoder_dense_layer.get_input_dimensions()
                    encoder_dense = None
                else:
                    encoder_architecture = encoder_dense_layer.architecture[:-1]
                    latent_input_size = encoder_dense_layer.architecture[-1]
                    encoder_dense = DenseLayers(
                        architecture=encoder_architecture,
                        input_size=encoder_dense_layer.input_size,
                        output_size=latent_input_size,
                    )
                latent_output_size = encoder_dense_layer.output_size
                latent_layer_mu = torch.nn.Linear(latent_input_size, latent_output_size)
                latent_layer_sigma = torch.nn.Linear(
                    latent_input_size, latent_output_size
                )
                latent_layer_vi = VAELinear(latent_input_size, latent_output_size)

                if encoder_dense is not None:
                    new_encoder.append(encoder_dense)
            else:
                new_encoder.append(layer)
        return (
            torch.nn.Sequential(*new_encoder),
            latent_layer_mu,
            latent_layer_sigma,
            latent_layer_vi,
        )

    def log_gaussian_loss_sigma_2_torch(self, y_pred, y_true, sigma_2):
        log_likelihood = (-((y_true - y_pred) ** 2) / (2 * sigma_2)) - (
            0.5 * torch.log(sigma_2)
        )
        return log_likelihood

    def kl_loss_prior_gaussian(self, weight, weight_mu, weight_sigma):
        q_variational_log_prob = self.log_gaussian_loss_sigma_2_torch(
            weight, weight_mu, weight_sigma
        )
        prior_log_prob = self.log_gaussian_loss_sigma_2_torch(
            weight, self.latent_layer_vi.prior_mu, self.latent_layer_vi.prior_sigma
        )
        kl_loss = (q_variational_log_prob - prior_log_prob).mean()

        return kl_loss

    def forward(self, x):
        encoded = self.encoder(x)
        latent_mu = self.latent_layer_mu(encoded)
        latent_sigma = self.latent_layer_sigma(encoded)

        # draw samples for weight and bias
        latent_sample = latent_mu + F.softplus(latent_sigma) * torch.randn_like(
            latent_mu
        )
        kl_loss = self.kl_loss_prior_gaussian(
            latent_sample, latent_mu, F.softplus(latent_sigma)
        )

        decoded_mu = self.decoder_mu(latent_sample)

        if self.decoder_sig_enabled:
            decoded_sig = self.decoder_sig(latent_sample)
            return decoded_mu, decoded_sig, kl_loss
        else:
            return decoded_mu, kl_loss


class VAE(BAE_VI):
    """
    For VAE, only the latent dimension layer is considered as probabilistic, and trained using VI
    """

    def __init__(
        self, *args, model_name="VAE", num_train_samples=5, beta=1.0, **kwargs
    ):
        self.num_train_samples = num_train_samples  # for training averaging
        BAE_VI.__init__(self, *args, model_name=model_name, **kwargs)

        # beta for KL weighting
        self.beta = beta

    def convert_autoencoder(self, autoencoder: Autoencoder):
        if autoencoder.decoder_sig_enabled:
            return VAE_Module(
                encoder=autoencoder.encoder,
                decoder_mu=autoencoder.decoder_mu,
                decoder_sig=autoencoder.decoder_sig,
            )
        else:
            return VAE_Module(
                encoder=autoencoder.encoder, decoder_mu=autoencoder.decoder_mu
            )

    def nll_kl_loss(self, autoencoder, x, y=None, mode="sigma"):
        # likelihood
        if y is None:
            y = x

        # forward sample of autoencoder
        if self.decoder_sigma_enabled:
            y_pred_mu, y_pred_sig, kl_loss = autoencoder(x)
        else:
            y_pred_mu, kl_loss = autoencoder(x)

        # get actual nll according to selected likelihood and mode
        if mode == "sigma":
            nll = self._nll(
                flatten_torch(y_pred_mu), flatten_torch(y), flatten_torch(y_pred_sig)
            )
        elif mode == "mu":
            nll = self._nll(
                flatten_torch(y_pred_mu), flatten_torch(y), autoencoder.log_noise
            )

        return nll, kl_loss

    def criterion(self, autoencoder, x, y=None, mode="sigma"):
        """
        Note that the kl_loss is for the probablistic latent layer,
        while prior_loss is for deterministic encoder and decoder(s)

        """
        # pass the data forward for num_train_samples times and obtain average loss
        for num_train_sample in range(self.num_train_samples):
            if num_train_sample == 0:
                nll, kl_loss = self.nll_kl_loss(autoencoder, x, y, mode)
            else:
                nll_temp, kl_loss_temp = self.nll_kl_loss(autoencoder, x, y, mode)
                nll = nll + nll_temp
                kl_loss = kl_loss + kl_loss_temp
        nll /= self.num_train_samples
        kl_loss /= self.num_train_samples

        # obtain mean of likelihood cost
        nll = nll.mean()

        # prior loss of encoder/decoder
        # note this doesn't include the latent layers
        # for kl loss already includes complexity cost due to prior on latent layers
        prior_loss_encoder = self.log_prior_loss(model=autoencoder.encoder).mean()
        prior_loss_decoder = self.log_prior_loss(model=autoencoder.decoder_mu).mean()
        prior_loss = prior_loss_encoder + prior_loss_decoder
        if self.decoder_sigma_enabled:
            prior_loss_decoder_sig = self.log_prior_loss(
                model=autoencoder.decoder_sig
            ).mean()
            prior_loss = prior_loss + prior_loss_decoder_sig

        # scale by beta
        kl_loss *= self.beta
        prior_loss *= self.weight_decay

        return nll + kl_loss + prior_loss

    def get_optimisers(self, autoencoder: Autoencoder, mode="mu", sigma_train="joint"):
        optimiser_list = self.get_optimisers_list(
            autoencoder, mode=mode, sigma_train=sigma_train
        )

        if sigma_train == "joint" or mode == "mu":
            optimiser_list.append({"params": autoencoder.latent_layer_vi.parameters()})
            optimiser_list.append({"params": autoencoder.latent_layer_mu.parameters()})
            optimiser_list.append(
                {"params": autoencoder.latent_layer_sigma.parameters()}
            )

        return torch.optim.Adam(optimiser_list, lr=self.learning_rate)

    def _get_mu_sigma_single(self, autoencoder, x):
        if self.decoder_sigma_enabled:
            y_mu, y_sigma, kl = autoencoder(x)
            del kl
        else:
            y_mu, kl_loss = autoencoder(x)
            del kl_loss
            y_sigma = torch.ones_like(y_mu)

        # convert to numpy
        y_mu = y_mu.detach().cpu().numpy()
        y_sigma = y_sigma.detach().cpu().numpy()
        log_noise = autoencoder.log_noise.detach().cpu().numpy()

        return flatten_np(y_mu), flatten_np(y_sigma), log_noise

    def predict_latent(self, x, transform_pca=True):
        x = self.convert_tensor(x)
        encoded = self.autoencoder.encoder(x)
        latent_mu = self.autoencoder.latent_layer_mu(encoded).detach().cpu().numpy()
        latent_sigma = (
            F.softplus(self.autoencoder.latent_layer_sigma(encoded))
            .detach()
            .cpu()
            .numpy()
        )

        # remove from gpu
        encoded = encoded.detach().cpu()  # detach
        del encoded

        # transform pca
        if transform_pca:
            pca = PCA(n_components=2)
            latent_pca_mu = pca.fit_transform(latent_mu)
            latent_pca_sig = pca.fit_transform(latent_sigma)
            return latent_pca_mu, latent_pca_sig
        else:
            return latent_mu, latent_sigma
