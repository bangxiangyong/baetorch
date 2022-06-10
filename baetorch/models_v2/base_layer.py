###TORCH BASE MODULES###
import copy

import numpy as np
import torch
from torch.nn import Parameter

from .dropout_layer import CustomDropout
from ..util.conv2d_util import (
    calc_flatten_conv2d_forward_pass,
    calc_flatten_conv2dtranspose_forward_pass,
    calc_flatten_conv1d_forward_pass,
    calc_flatten_conv1dtranspose_forward_pass,
    calc_required_padding,
    convert_tuple_conv2d_params,
)
from ..util.dense_util import parse_architecture_string_v2
from ..util.misc import parse_activation


def create_base_layer(base, transpose, input_size, output_size, bias, **base_params):
    # handle base layers : either conv2d, conv1d or linear.
    if base == "conv2d":
        if transpose:
            base_layer = torch.nn.ConvTranspose2d(
                in_channels=input_size,
                out_channels=output_size,
                bias=bias,
                **base_params,
            )
        else:
            base_layer = torch.nn.Conv2d(
                in_channels=input_size,
                out_channels=output_size,
                bias=bias,
                **base_params,
            )
    elif base == "conv1d":
        if transpose:
            base_layer = torch.nn.ConvTranspose1d(
                in_channels=input_size,
                out_channels=output_size,
                bias=bias,
                **base_params,
            )
        else:
            base_layer = torch.nn.Conv1d(
                in_channels=input_size,
                out_channels=output_size,
                bias=bias,
                **base_params,
            )
    elif base == "linear":
        base_layer = torch.nn.Linear(
            in_features=input_size, out_features=output_size, bias=bias, **base_params
        )
    else:
        raise NotImplemented("Invalid base layer selected")

    return base_layer


def create_block(
    input_size,
    output_size,
    base="conv2d",
    order=["base", "norm", "activation"],
    activation="leakyrelu",
    norm: str = "none",
    transpose=False,
    bias=False,
    dropout=0,
    dropout_n=1,
    create_base_layer_func=create_base_layer,
    torch_wrapper=torch.nn.Sequential,
    se_block=False,
    self_att=False,
    self_att_transpose_only=False,
    **base_params,
):
    """
    Creates a block which consist of (1) base layer (e.g linear or convolution), (2) batch norm, and (3) activation.
    The order of layers can be explicitly specified via "order".

    transpose: Boolean
        Only applicable to convolution layers which are common for decoders.
    """
    # handle base layers : either conv2d, conv1d or linear.
    base_layer = create_base_layer_func(
        base, transpose, input_size, output_size, bias, **base_params
    )

    # handle batch normalisation which is also dependent on the base layer
    # get indices of "base" and "norm" in order (either "norm" is after/before the base layer)
    # required to get the correct size for the norm input size
    if norm:
        norm_after_base = False
        if "norm" in order:
            norm_order_id = order.index("norm")
            base_order_id = order.index("base")
            if norm_order_id > base_order_id:
                norm_after_base = True

        # if order of "norm" layer is not specified, we assume it at the end.
        elif "norm" not in order:
            order.append("norm")
            norm_after_base = True
        batch_norm_size = output_size if norm_after_base else input_size

        # handle exact type of norm layer
        # either batch / layer/ or instance norm
        if norm == "batch":
            bn_momentum = 0.01
            track_running_stats = False
            if base == "conv2d":
                norm_layer = torch.nn.BatchNorm2d(
                    batch_norm_size,
                    momentum=bn_momentum,
                    track_running_stats=track_running_stats,
                )
            if base == "conv1d":
                norm_layer = torch.nn.BatchNorm1d(
                    batch_norm_size,
                    momentum=bn_momentum,
                    track_running_stats=track_running_stats,
                )
            if base == "linear":
                norm_layer = torch.nn.BatchNorm1d(
                    batch_norm_size,
                    momentum=bn_momentum,
                    track_running_stats=track_running_stats,
                )

        elif norm == "instance":
            norm_layer = torch.nn.GroupNorm(
                batch_norm_size, batch_norm_size
            )  # Instance Norm

        elif norm == "layer":
            if base == "linear":
                norm_layer = torch.nn.GroupNorm(1, batch_norm_size)  # Layer Norm

            else:
                norm_layer = torch.nn.GroupNorm(1, batch_norm_size)  # Layer Norm

        elif norm == "weight":
            base_layer = torch.nn.utils.weight_norm(base_layer)

    # set default norm
    else:
        norm = "none"

    # start appending actual layers into the block based on given order
    # returns a torch sequential of the layers
    block = []

    for layer_type in order:
        if layer_type == "base":
            block.append(base_layer)

        elif layer_type == "activation":
            activation_layer = parse_activation(activation=activation)
            if activation_layer is not None:
                block.append(activation_layer)

        # ignore adding norm layer if it is "weight" or "none"
        # since weight layer is a wrapper to the base layer
        elif layer_type == "norm" and norm != "weight" and norm != "none":
            block.append(norm_layer)

    # handle adding dropout
    if dropout is not None and dropout > 0:
        block.append(CustomDropout(drop_p=dropout, n_samples=dropout_n))

    if se_block and base == "conv2d":
        block.append(SE_Block(output_size, r=16))

    return torch_wrapper(*block)


def create_twin_block(
    input_size,
    output_size,
    base="conv2d",
    order=["base", "norm", "activation"],
    activation="leakyrelu",
    norm: str = "none",
    transpose=False,
    bias=False,
    twin_params={},
    create_block_func=create_block,
    **base_params,
):
    # explicitly specify parameters of the twin
    twin_params_ = {
        "base": base,
        "order": order,
        "activation": activation,
        "norm": norm,
        "transpose": transpose,
        "bias": bias,
    }
    twin_params_.update(twin_params)
    twin_params_.update(base_params)

    # create main block
    main_block = create_block_func(
        input_size=input_size,
        output_size=output_size,
        base=base,
        order=order,
        activation=activation,
        norm=norm,
        bias=bias,
        transpose=transpose,
        **base_params,
    )

    # create twin block
    twin_block = create_block_func(
        input_size=input_size,
        output_size=output_size,
        **twin_params_,
    )

    # combine both as a twin output module
    return TwinOutputModule([main_block, twin_block])


def create_conv_chain(
    input_dim=32,
    conv_channels=[1, 32, 64],
    conv_kernel=3,
    conv_stride=1,
    base="conv2d",
    order=["base", "norm", "activation"],
    activation="leakyrelu",
    norm="none",
    bias=False,
    last_activation=None,
    last_norm=None,
    transpose=False,
    twin_output=False,
    twin_params={},
    dropout=0,
    last_dropout=None,
    create_block_func=create_block,
    *block_args,
    **block_kwargs,
):
    """
    Returns a list of convolutional blocks.

    For enabling twin output, first set twin_output to True and supply the "activation" and "norm" values.
    """

    # ====House Keeping===
    if last_activation is None:
        last_activation = activation
    if last_norm is None:
        last_norm = norm

    conv_dim = 2 if base == "conv2d" else 1

    # convert into a list if conv2d is used
    if conv_dim == 2:
        input_dim = convert_tuple_conv2d_params(input_dim)

    conv_padding, output_padding = calc_required_padding(
        input_dim_init=input_dim,
        kernels=conv_kernel,
        strides=conv_stride,
        verbose=True,
        conv_dim=conv_dim,
    )

    conv_channels_ = copy.copy(conv_channels)
    conv_kernel_ = copy.copy(conv_kernel)
    conv_padding_ = copy.copy(conv_padding)
    conv_stride_ = copy.copy(conv_stride)

    if transpose:
        conv_channels_.reverse()
        conv_kernel_.reverse()
        conv_padding_.reverse()
        conv_stride_.reverse()

    # handle last dropout param
    if last_dropout is None:
        last_dropout = dropout

    # ===Create a block for each channel===
    last_channel_id = len(conv_channels_) - 2
    chain = []
    for channel_id, num_channels in enumerate(conv_channels_[:-1]):
        in_channels = conv_channels_[channel_id]
        out_channels = conv_channels_[channel_id + 1]
        output_padding_param = (
            {"output_padding": output_padding[channel_id]} if transpose else {}
        )
        block_kwargs.update(output_padding_param)
        # handle if it is the last layer
        # specifically, last_norm and last_activation will be used here
        if channel_id != last_channel_id:
            chain.append(
                create_block_func(
                    input_size=in_channels,
                    output_size=out_channels,
                    base=base,
                    order=order,
                    activation=activation,
                    norm=norm,
                    bias=bias,
                    transpose=transpose,
                    kernel_size=conv_kernel_[channel_id],
                    stride=conv_stride_[channel_id],
                    padding=conv_padding_[channel_id],
                    dropout=dropout,
                    *block_args,
                    **block_kwargs,
                )
            )
        # last layer handle differently
        # to specify separate norm and activation
        else:
            # single output
            if not twin_output:
                chain.append(
                    create_block_func(
                        input_size=in_channels,
                        output_size=out_channels,
                        base=base,
                        order=order,
                        activation=last_activation,
                        norm=last_norm,
                        bias=bias,
                        transpose=transpose,
                        kernel_size=conv_kernel_[channel_id],
                        stride=conv_stride_[channel_id],
                        padding=conv_padding_[channel_id],
                        dropout=last_dropout,
                        *block_args,
                        **block_kwargs,
                    )
                )
            else:  # twin output
                chain.append(
                    create_twin_block(
                        input_size=in_channels,
                        output_size=out_channels,
                        base=base,
                        order=order,
                        activation=last_activation,
                        norm=last_norm,
                        bias=bias,
                        transpose=transpose,
                        kernel_size=conv_kernel_[channel_id],
                        stride=conv_stride_[channel_id],
                        padding=conv_padding_[channel_id],
                        twin_params=twin_params,
                        dropout=last_dropout,
                        create_block_func=create_block_func,
                        *block_args,
                        **block_kwargs,
                    )
                )
    return chain


class TwinOutputModule(torch.nn.ModuleList):
    def forward(self, x):
        final_out = [layer(x) for layer in self]
        return final_out


def create_linear_chain(
    architecture=[],
    order=["base", "norm", "activation"],
    activation="leakyrelu",
    norm="none",
    bias=False,
    last_activation=None,
    last_norm=None,
    transpose=False,
    twin_output=False,
    twin_params={},
    dropout=0,
    last_dropout=None,
    base="linear",
    create_block_func=create_block,
    *block_args,
    **block_kwargs,
):
    """
    Returns a list of linear blocks.
    """
    # check for zero-sized nodes
    if any(node == 0 for node in architecture):
        raise ValueError(
            "Linear layer's node size cannot be zero! Got architecture string "
            + str(architecture)
        )

    if last_activation is None:
        last_activation = activation
    if last_norm is None:
        last_norm = norm

    architecture_ = copy.copy(architecture)
    if transpose:
        architecture_ = architecture_[::-1]

    # handle last dropout param
    if last_dropout is None and dropout > 0:
        last_dropout = dropout

    # ===Create a block for each channel===
    last_channel_id = len(architecture_) - 2
    chain = []
    for channel_id, num_channels in enumerate(architecture_[:-1]):
        in_channels = architecture_[channel_id]
        out_channels = architecture_[channel_id + 1]

        # handle if it is the last layer
        # specifically, last_norm and last_activation will be used here
        if channel_id != last_channel_id:
            chain.append(
                create_block_func(
                    input_size=in_channels,
                    output_size=out_channels,
                    base="linear",
                    order=order,
                    activation=activation,
                    norm=norm,
                    bias=bias,
                    dropout=dropout,
                    *block_args,
                    **block_kwargs,
                )
            )
        else:
            if not twin_output:
                chain.append(
                    create_block_func(
                        input_size=in_channels,
                        output_size=out_channels,
                        base="linear",
                        order=order,
                        activation=last_activation,
                        norm=last_norm,
                        bias=bias,
                        dropout=last_dropout,
                        *block_args,
                        **block_kwargs,
                    )
                )
            else:
                chain.append(
                    create_twin_block(
                        input_size=in_channels,
                        output_size=out_channels,
                        base="linear",
                        order=order,
                        activation=last_activation,
                        norm=last_norm,
                        bias=bias,
                        twin_params=twin_params,
                        dropout=last_dropout,
                        create_block_func=create_block_func,
                        *block_args,
                        **block_kwargs,
                    )
                )

    return chain


def get_conv_latent_shapes(conv_chain, *input_dim):
    """
    Given a conv chain, creates dummy data to forward pass it to get the shapes of outputs.
    Used for determining shapes of latent.

    Returns the full and flattened version of the latent shapes.
    """

    # determine the input channel
    first_block = conv_chain[0]
    for layer in first_block:
        if (
            isinstance(layer, torch.nn.Conv2d)
            or isinstance(layer, torch.nn.Conv1d)
            or isinstance(layer, torch.nn.ConvTranspose2d)
            or isinstance(layer, torch.nn.ConvTranspose1d)
        ):
            first_conv_channel = layer.in_channels

    # create dummy data
    dummy_data = torch.randn((1, first_conv_channel, *input_dim))
    latent_shapes = []
    for block in conv_chain:
        dummy_data = block(dummy_data)

        # handle VI which returns a list (with second item being KL div)
        if isinstance(dummy_data, list):
            latent_shapes.append(list(dummy_data[0].shape[1:]))
        else:
            latent_shapes.append(list(dummy_data.shape[1:]))
    latent_shapes = np.array(latent_shapes)
    flatten_latent_shapes = np.product(latent_shapes, 1)

    return latent_shapes, flatten_latent_shapes


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
                if norm and channel_id != (len(self.conv_architecture) - 2):
                    layer_list.append(torch.nn.BatchNorm2d(out_channels, momentum=0.01))

                elif last_norm and channel_id == (len(self.conv_architecture) - 2):
                    layer_list.append(torch.nn.BatchNorm2d(out_channels, momentum=0.01))

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
        **kwargs,
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

        # parse architecture string and add
        self.layers = self.init_layers(
            layer_type, activation=activation, last_activation=last_activation
        )
        self.log_noise_size = log_noise_size
        self.set_log_noise(self.init_log_noise, log_noise_size=log_noise_size)
        self.model_kwargs = kwargs
        self.norm = norm

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
            layers = parse_architecture_string_v2(
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
            layers = parse_architecture_string_v2(
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
