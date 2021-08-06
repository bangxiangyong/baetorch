import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.conv import Conv2d, ConvTranspose2d, ConvTranspose1d, Conv1d

from baetorch.baetorch.models_v2.base_autoencoder import AutoencoderModule
from baetorch.baetorch.models_v2.base_layer import (
    create_block,
    TwinOutputModule,
    create_linear_chain,
    create_conv_chain,
)


class VI_AutoencoderModule(AutoencoderModule):
    """
    Required to specifically handle skip connection for VI.
    """

    def forward(self, x):
        if self.skip:
            return self.forward_skip(x)
        else:
            for block in self.encoder:
                # handle variational block
                if isinstance(block, VariationalBlock):
                    x = block(x)
                else:
                    x = [block(x[0]), x[1]]

            for block in self.decoder:
                # handle variational block
                if isinstance(block, VariationalBlock):
                    x = block(x)
                else:
                    x = [block(x[0]), x[1]]

            # unpack twin output
            if self.twin_output:
                x = [[x[0][0][0], x[0][1][0]], x[1] + x[0][0][1] + x[0][1][1]]
            return x

    def forward_skip(self, x):
        # implement skip connections from encoder to decoder
        enc_outs = []

        # collect encoder outputs
        for enc_i, block in enumerate(self.encoder):
            # handle variational block forward pass
            if isinstance(block, VariationalBlock):
                x = block(x)
            else:
                x = [block(x[0]), x[1]]

            # collect output of encoder-blocks if it is not the last, and also
            # a valid Sequential block (unlike flatten/reshape)
            if enc_i != self.num_enc_blocks - 1 and isinstance(
                block, torch.nn.Sequential
            ):
                enc_outs.append(x[0])

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
                x[0] += enc_outs[skip_i]
                skip_i += 1

            # handle variational block forward pass
            if isinstance(block, VariationalBlock):
                x = block(x)
            else:
                x = [block(x[0]), x[1]]

        # unpack twin output
        if self.twin_output:
            x = [[x[0][0][0], x[0][1][0]], x[1] + x[0][0][1] + x[0][1][1]]

        return x

    def init_create_chain_func(self):
        """
        Override this to provide custom linear or conv chain functions.
        """

        # define method to create type of chain
        self.create_chain_func = {
            "linear": create_vlinear_chain,
            "conv2d": create_vconv_chain,
            "conv1d": create_conv_chain,
        }


def create_vlinear_chain(**params):
    return create_linear_chain(create_block_func=create_variational_block, **params)


def create_vconv_chain(**params):
    return create_conv_chain(create_block_func=create_variational_block, **params)


def create_variational_layer(
    base, transpose, input_size, output_size, bias, **base_params
):
    # handle base layers : either conv2d, conv1d or linear.
    if base == "conv2d":
        if transpose:
            base_layer = VariationalConv2DTranspose(
                in_channels=input_size,
                out_channels=output_size,
                bias=bias,
                **base_params,
            )
        else:
            base_layer = VariationalConv2D(
                in_channels=input_size,
                out_channels=output_size,
                bias=bias,
                **base_params,
            )
    elif base == "conv1d":
        if transpose:
            base_layer = VariationalConv2DTranspose(
                in_channels=input_size,
                out_channels=output_size,
                bias=bias,
                **base_params,
            )
        else:
            base_layer = VariationalConv1D(
                in_channels=input_size,
                out_channels=output_size,
                bias=bias,
                **base_params,
            )
    elif base == "linear":
        base_layer = VariationalLinear(
            input_size=input_size, output_size=output_size, bias=bias, **base_params
        )
    else:
        raise NotImplemented("Invalid base layer selected")

    return base_layer


class VariationalBlock(torch.nn.Sequential):
    def forward(self, x):
        for id_, layer in enumerate(self):
            # if received a list i.e from a previous variational layer
            # unpack it to kl_loss
            if isinstance(x, list):
                x, kl_loss = x
                add_kl = True
            else:
                add_kl = False
            if (
                isinstance(layer, VariationalLinear)
                or isinstance(layer, VariationalConv1D)
                or isinstance(layer, VariationalConv1DTranspose)
                or isinstance(layer, VariationalConv2D)
                or isinstance(layer, VariationalConv2DTranspose)
            ):
                x, kl_loss_new = layer(x)
                if add_kl:
                    kl_loss_new += kl_loss
            else:
                x = layer(x)

        return [x, kl_loss_new]


def create_variational_block(**params):
    vblock = create_block(
        create_base_layer_func=create_variational_layer,
        torch_wrapper=VariationalBlock,
        **params,
    )
    return vblock


class VariationalLinear(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        bias=False,
        prior_mu=0.0,
        prior_sigma_1=1.0,
        prior_sigma_2=0.1,
        prior_pi=0.5,
    ):
        super(VariationalLinear, self).__init__()
        self.weight_mu = Parameter(torch.Tensor(output_size, input_size))
        self.weight_sigma = Parameter(torch.Tensor(output_size, input_size))
        self.bias = bias
        if self.bias is None or not self.bias:
            self.enable_bias = False
        else:
            self.enable_bias = True
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

            if self.enable_bias:
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
        if self.enable_bias:
            bias = self.bias_sample()
        else:
            bias = None
        kl_loss = self.kl_loss_prior_mixture(
            weight, self.weight_mu, F.softplus(self.weight_sigma)
        )
        y = F.linear(x, weight, bias)

        return [y, kl_loss]


class VariationalBaseConv:
    def __init__(
        self, prior_mu=0.0, prior_sigma_1=1.0, prior_sigma_2=0.1, prior_pi=0.5
    ):
        self.weight_mu = Parameter(torch.Tensor(*self.weight.shape))
        self.weight_sigma = Parameter(torch.Tensor(*self.weight.shape))
        if self.bias is None:
            self.enable_bias = False
        else:
            self.enable_bias = True
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

            if self.enable_bias:
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
        bias = self.bias_sample() if self.enable_bias else None
        kl_loss = self.kl_loss_prior_mixture(
            weight, self.weight_mu, F.softplus(self.weight_sigma)
        )
        y = F.conv2d(
            x, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        return [y, kl_loss]


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
        bias = self.bias_sample() if self.enable_bias else None
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
        return [y, kl_loss]


class VariationalConv1D(Conv1d, VariationalBaseConv):
    def __init__(self, **kwargs):
        Conv1d.__init__(self, **kwargs)
        VariationalBaseConv.__init__(self)
        self.reset_parameters()

    def reset_parameters(self):
        Conv1d.reset_parameters(self)
        VariationalBaseConv.reset_parameters(self)

    def forward(self, x):
        # draw samples for weight and bias
        weight = self.weight_sample()
        bias = self.bias_sample() if self.enable_bias else None
        kl_loss = self.kl_loss_prior_mixture(
            weight, self.weight_mu, F.softplus(self.weight_sigma)
        )
        y = F.conv1d(
            x, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        return [y, kl_loss]


class VariationalConv1DTranspose(ConvTranspose1d, VariationalBaseConv):
    def __init__(self, **kwargs):
        ConvTranspose1d.__init__(self, **kwargs)
        VariationalBaseConv.__init__(self)
        self.reset_parameters()

    def reset_parameters(self):
        ConvTranspose1d.reset_parameters(self)
        VariationalBaseConv.reset_parameters(self)

    def forward(self, x):
        # draw samples for weight and bias
        weight = self.weight_sample()
        bias = self.bias_sample() if self.enable_bias else None
        kl_loss = self.kl_loss_prior_mixture(
            weight, self.weight_mu, F.softplus(self.weight_sigma)
        )
        y = F.conv_transpose1d(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )
        return [y, kl_loss]
