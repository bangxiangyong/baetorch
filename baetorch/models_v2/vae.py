import torch
from torch.nn import Flatten, Parameter
import torch.nn.functional as F

from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.base_autoencoder import AutoencoderModule
from baetorch.baetorch.models_v2.base_layer import (
    get_conv_latent_shapes,
    flatten_torch,
    TwinOutputModule,
)
from baetorch.baetorch.models_v2.vi_layer import VariationalLinear


class VAEModule(AutoencoderModule):
    def __init__(self, **params):
        super(VAEModule, self).__init__(**params)

        # add the VAE Latent sampling layer
        self.latent_layer = VAE_Latent(self.use_cuda)

    def instantiate_encoder(self, chain_params=[{"base": "linear"}]):
        # instantiate the first chain
        # VAE: twin_output is only enabled on encoder's last layer
        encoder = self.create_chain_func[chain_params[0]["base"]](
            twin_output=False if self.conv_linear_type else True, **chain_params[0]
        )

        # handle conv-linear type
        # by adding Flatten() layer
        # and making sure the linear input matches the flattened conv layer
        if self.conv_linear_type:
            conv_params = chain_params[0].copy()
            linear_params = chain_params[1].copy()

            inp_dim = (
                [conv_params["input_dim"]]
                if isinstance(conv_params["input_dim"], int)
                else conv_params["input_dim"]
            )

            # get the flattened latent shapes
            (
                self.conv_latent_shapes,
                self.flatten_latent_shapes,
            ) = get_conv_latent_shapes(encoder, *inp_dim)

            linear_params.update(
                {
                    "architecture": [self.flatten_latent_shapes[-1]]
                    + linear_params["architecture"]
                }
            )
            self.dec_params[0].update(linear_params)  # update decoder params

            # append flatten layer
            encoder.append(Flatten())

            # append linear chain
            encoder = encoder + self.create_chain_func[linear_params["base"]](
                twin_output=True, **linear_params
            )

        self.encoder = torch.nn.Sequential(*encoder)

    def forward(self, x):
        if self.skip:
            return self.forward_skip(x)
        else:
            enc_mu, enc_sig = self.encoder(x)
            x, kl_loss = self.latent_layer(enc_mu, enc_sig)
            x = self.decoder(x)
            return [x, kl_loss]

    def forward_skip(self, x):
        # implement skip connections from encoder to decoder
        enc_outs = []

        # collect encoder outputs
        for enc_i, block in enumerate(self.encoder):
            x = block(x)

            # collect output of encoder-blocks if it is not the last, and also
            # a valid Sequential block (unlike flatten/reshape)
            if enc_i != self.num_enc_blocks - 1 and isinstance(
                block, torch.nn.Sequential
            ):
                enc_outs.append(x)

        # VAE : forward latent sample and unpack kl_loss
        enc_mu, enc_sig = x
        x, kl_loss = self.latent_layer(enc_mu, enc_sig)

        # reverse the order to add conveniently to the decoder-blocks outputs
        enc_outs.reverse()

        # now run through decoder-blocks
        # we apply the encoder-blocks output to the decoder blocks' inputs.
        # while ignoring the first decoder block
        skip_i = 0
        for dec_i, block in enumerate(self.decoder):
            if (
                dec_i != 0
                and isinstance(block, torch.nn.Sequential)
                or isinstance(block, TwinOutputModule)
            ):
                x += enc_outs[skip_i]
                skip_i += 1
            x = block(x)

        return [x, kl_loss]


class VAE_Latent(torch.nn.Module):
    def __init__(self, use_cuda=False):
        super(VAE_Latent, self).__init__()
        self.first_time = True
        self.use_cuda = use_cuda

    def init_params(self, latent_size, prior_mu=0.0, prior_sigma=1.0):
        self.latent_size = latent_size
        self.latent_mu = Parameter(torch.Tensor(latent_size))
        self.latent_sigma = Parameter(torch.Tensor(latent_size))
        self.prior_mu = Parameter(torch.FloatTensor([prior_mu]), requires_grad=False)
        self.prior_sigma = Parameter(
            torch.FloatTensor([prior_sigma]), requires_grad=False
        )

    def log_gaussian_loss_sigma_2_torch(self, y_pred, y_true, sigma_2):
        log_likelihood = (-((y_true - y_pred) ** 2) / (2 * sigma_2)) - (
            0.5 * torch.log(sigma_2)
        )
        return log_likelihood

    def kl_loss_prior_gaussian(self, latent, latent_mu, latent_sigma):
        q_variational_log_prob = self.log_gaussian_loss_sigma_2_torch(
            latent, latent_mu, latent_sigma
        )
        prior_log_prob = self.log_gaussian_loss_sigma_2_torch(
            latent, self.prior_mu, self.prior_sigma
        )
        kl_loss = (q_variational_log_prob - prior_log_prob).mean()

        return kl_loss

    def latent_sample(self):
        # depracated
        latent = self.latent_mu + F.softplus(self.latent_sigma) * torch.randn_like(
            self.latent_mu
        )
        return latent

    def latent_sample_v2(self, latent_mu, latent_sigma):
        latent = latent_mu + F.softplus(latent_sigma) * torch.randn_like(latent_mu)
        return latent

    def forward(self, latent_mu, latent_sigma):
        # check if reshape is needed
        if len(latent_mu.shape) > 2:
            self.reshape_size = list(latent_mu.shape)
            latent_mu = flatten_torch(latent_mu)
            latent_sigma = flatten_torch(latent_sigma)
        else:
            self.reshape_size = None

        # init size of weight_mu and weight_sigma for first time
        if self.first_time:
            latent_size = latent_mu.shape[-1]
            self.init_params(latent_size)
            if self.use_cuda:
                self.cuda()
            self.first_time = False

        # draw samples for weight and bias
        latent = self.latent_sample_v2(latent_mu, latent_sigma)

        kl_loss = self.kl_loss_prior_gaussian(
            latent, latent_mu, F.softplus(latent_sigma)
        )

        # reshape to original size if needed
        if self.reshape_size is not None:
            latent = latent.view(*self.reshape_size)
        return [latent, kl_loss]


# class VAE
class VAE(BAE_VI):
    def __init__(self, num_train_samples=1, num_test_samples=100, **params):
        # store variables specific to mc_dropout
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

        # override unused params
        params.update({"anchored": False, "sparse_scale": 0})
        super(BAE_VI, self).__init__(
            model_type="stochastic", AE_Module=VAEModule, **params
        )

        # forward activation loss
        self.activ_loss = True
