###TORCH BASE MODULES###
import copy

import torch
from torch.nn import Parameter
import numpy as np

from ..util.conv2d_util import (
    calc_flatten_conv2d_forward_pass,
    calc_flatten_conv2dtranspose_forward_pass,
    calc_flatten_conv1d_forward_pass,
    calc_flatten_conv1dtranspose_forward_pass,
    calc_required_padding,
    convert_tuple_conv2d_params,
)
from ..util.dense_util import parse_architecture_string, parse_architecture_string_v2
from ..util.misc import parse_activation


class ConvLayers(torch.nn.Module):
    def __init__(
        self,
        input_dim=28,
        conv_architecture=[1, 32, 64],
        conv_kernel=3,
        conv_stride=1,
        conv_padding=2,
        reverse_params=True,
        mpool_kernel=2,
        mpool_stride=2,
        output_padding=[],
        use_cuda=False,
        activation="relu",
        upsampling=False,
        last_activation="sigmoid",
        layer_type=[torch.nn.Conv2d, torch.nn.ConvTranspose2d],
        conv_dim=2,
        norm=True,
        bias=False,
        add_se=False,
        last_norm=True,
    ):
        super(ConvLayers, self).__init__()
        self.layers = []
        self.use_cuda = use_cuda
        self.add_se = add_se
        self.norm = norm
        self.bias = bias
        self.upsampling = upsampling
        self.conv_architecture = copy.copy(conv_architecture)
        self.conv_kernel, self.conv_stride, self.conv_padding = (
            copy.copy(conv_kernel),
            copy.copy(conv_stride),
            copy.copy(conv_padding),
        )
        self.conv_kernel = self.convert_int_to_list(
            self.conv_kernel, len(self.conv_architecture) - 1
        )
        self.conv_stride = self.convert_int_to_list(
            self.conv_stride, len(self.conv_architecture) - 1
        )

        if conv_dim == 2:
            self.input_dim = convert_tuple_conv2d_params(input_dim)
        else:
            self.input_dim = input_dim

        self.activation = activation
        self.last_activation = last_activation
        self.conv_dim = conv_dim

        # forward and deconvolutional layer type
        self.conv_layer_type = layer_type[0]
        self.conv_trans_layer_type = layer_type[1]

        if len(output_padding) == 0:
            self.conv_padding, self.output_padding = calc_required_padding(
                input_dim_init=input_dim,
                kernels=conv_kernel,
                strides=conv_stride,
                verbose=True,
                conv_dim=self.conv_dim,
            )
        else:
            self.conv_padding = self.convert_int_to_list(
                self.conv_padding, len(self.conv_architecture) - 1
            )
            self.output_padding = output_padding
        if isinstance(self.output_padding, np.ndarray):
            self.output_padding = tuple(self.output_padding)

        self.last_activation = last_activation

        if self.upsampling and reverse_params:
            self.conv_architecture.reverse()
            self.conv_kernel.reverse()
            self.conv_padding.reverse()
            self.conv_stride.reverse()

        # create sequence of conv2d layers and max pools
        for channel_id, num_channels in enumerate(self.conv_architecture):
            if channel_id != (len(self.conv_architecture) - 1):
                in_channels = self.conv_architecture[channel_id]
                out_channels = self.conv_architecture[channel_id + 1]

                # activation of last layer
                if channel_id == (len(self.conv_architecture) - 2) and self.upsampling:
                    activation = parse_activation(last_activation)
                else:
                    activation = parse_activation(activation)

                # standard convolutional
                if self.upsampling == False:
                    layer_list = [
                        self.conv_layer_type(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=self.conv_kernel[channel_id],
                            stride=self.conv_stride[channel_id],
                            padding=self.conv_padding[channel_id],
                            bias=bias,
                        )
                    ]

                # deconvolutional
                else:
                    layer_list = [
                        self.conv_trans_layer_type(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=self.conv_kernel[channel_id],
                            stride=self.conv_stride[channel_id],
                            padding=self.conv_padding[channel_id],
                            output_padding=self.output_padding[channel_id],
                            bias=bias,
                        ),
                    ]

                # handle batch norm
                # if norm and channel_id != (len(self.conv_architecture) - 2):
                #     layer_list.append(torch.nn.BatchNorm2d(out_channels))
                #
                # elif last_norm and channel_id == (len(self.conv_architecture) - 2):
                #     layer_list.append(torch.nn.BatchNorm2d(out_channels))

                # handle activation
                if activation is not None:
                    layer_list.append(activation)

                # add se block to last layer
                if channel_id == (len(self.conv_architecture) - 2) and self.add_se:
                    layer_list.append(SE_Block(self.conv_architecture[-1], r=16))
                layer = torch.nn.Sequential(*layer_list)

                self.layers.append(layer)

        self.layers = torch.nn.ModuleList(self.layers)

        if self.use_cuda:
            self.layers.cuda()

    def get_input_dimensions(self, flatten=True):
        if flatten:
            if self.conv_dim == 2:
                return (self.input_dim[0] * self.input_dim[1]) * self.conv_architecture[
                    0
                ]
            else:
                return self.input_dim * self.conv_architecture[0]
        else:
            return (self.conv_architecture[0], self.input_dim)

    def get_output_dimensions(self, input_dim=None, flatten=True):
        if input_dim is None:
            input_dim = self.input_dim
        if self.conv_dim == 2:
            if self.upsampling == False:
                return calc_flatten_conv2d_forward_pass(
                    input_dim,
                    channels=self.conv_architecture,
                    strides=self.conv_stride,
                    kernels=self.conv_kernel,
                    paddings=self.conv_padding,
                    flatten=flatten,
                )
            else:
                return calc_flatten_conv2dtranspose_forward_pass(
                    input_dim,
                    channels=self.conv_architecture,
                    strides=self.conv_stride[::-1],
                    kernels=self.conv_kernel[::-1],
                    paddings=self.conv_padding[::-1],
                    output_padding=self.output_padding,
                    flatten=flatten,
                )
        else:
            if self.upsampling == False:
                return calc_flatten_conv1d_forward_pass(
                    input_dim,
                    channels=self.conv_architecture,
                    strides=self.conv_stride,
                    kernels=self.conv_kernel,
                    paddings=self.conv_padding,
                    flatten=flatten,
                )
            else:
                return calc_flatten_conv1dtranspose_forward_pass(
                    input_dim,
                    channels=self.conv_architecture,
                    strides=self.conv_stride[::-1],
                    kernels=self.conv_kernel[::-1],
                    paddings=self.conv_padding[::-1],
                    output_padding=self.output_padding,
                    flatten=flatten,
                )

    def convert_int_to_list(self, int_param, num_replicate):
        """
        To handle integer passed as param, creates replicate of list
        """
        if isinstance(int_param, int):
            return [int_param] * num_replicate
        else:
            return int_param

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Conv2DLayers(ConvLayers):
    def __init__(self, **kwargs):
        super(Conv2DLayers, self).__init__(
            **kwargs, layer_type=[torch.nn.Conv2d, torch.nn.ConvTranspose2d], conv_dim=2
        )


class Conv1DLayers(ConvLayers):
    def __init__(self, **kwargs):
        super(Conv1DLayers, self).__init__(
            **kwargs, layer_type=[torch.nn.Conv1d, torch.nn.ConvTranspose1d], conv_dim=1
        )


class DenseLayers(torch.nn.Module):
    def __init__(
        self,
        input_size=1,
        output_size=1,
        architecture=["d1", "d1"],
        activation="relu",
        use_cuda=False,
        init_log_noise=1e-3,
        last_activation="none",
        layer_type=torch.nn.Linear,
        log_noise_size=1,
        norm=True,
        last_norm=True,
        **kwargs
    ):
        super(DenseLayers, self).__init__()
        self.architecture = architecture
        self.use_cuda = use_cuda
        self.input_size = input_size
        self.output_size = output_size
        self.init_log_noise = init_log_noise
        self.last_activation = last_activation
        self.activation = activation
        self.last_norm = last_norm
        self.norm = norm

        # parse architecture string and add
        self.layers = self.init_layers(
            layer_type, activation=activation, last_activation=last_activation
        )
        self.log_noise_size = log_noise_size
        self.set_log_noise(self.init_log_noise, log_noise_size=log_noise_size)
        self.model_kwargs = kwargs

    def get_input_dimensions(self, flatten=True):
        return self.input_size

    def set_log_noise(self, log_noise, log_noise_size=1):
        if log_noise_size == 0:  # log noise is turned off
            self.log_noise = Parameter(torch.FloatTensor([[0.0]]), requires_grad=False)

        else:
            self.log_noise = Parameter(
                torch.FloatTensor([[np.log(log_noise)] * log_noise_size])
            )

    def init_layers(
        self,
        layer_type=torch.nn.Linear,
        architecture=None,
        input_size=None,
        output_size=None,
        activation=None,
        last_activation=None,
    ):
        # resort to default input_size
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

        if activation is None:
            activation = self.activation
        else:
            self.activation = activation

        # resort to default architecture
        if architecture is None:
            layers = parse_architecture_string(
                input_size,
                output_size,
                self.architecture,
                layer_type=layer_type,
                activation=activation,
                last_activation=last_activation,
                norm=self.norm,
                last_norm=self.last_norm,
            )
        else:
            layers = parse_architecture_string(
                input_size,
                output_size,
                architecture,
                layer_type=layer_type,
                activation=activation,
                last_activation=last_activation,
                norm=self.norm,
                last_norm=self.last_norm,
            )

        if self.use_cuda:
            layers = torch.nn.ModuleList(layers).cuda()
        else:
            layers = torch.nn.ModuleList(layers)
        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def flatten_torch(x):
    x = x.reshape(x.size()[0], -1)
    return x


def flatten_np(x):
    x = x.reshape(x.shape[0], -1)
    return x


class Flatten(torch.nn.Module):
    def forward(self, x):
        if isinstance(x, tuple):
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
        if isinstance(x, tuple):
            y = x[0].view(x[0].size()[0], *tuple(self.size))
            return y, x[1:]
        else:
            y = x.view(x.size()[0], *tuple(self.size))
            return y


class SE_Block(torch.nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = torch.nn.AdaptiveAvgPool2d(1)
        ex_channel = c // r
        if ex_channel == 0:
            ex_channel = 1
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(c, ex_channel, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(ex_channel, c, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
