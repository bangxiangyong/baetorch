import numpy as np
import copy
from itertools import combinations_with_replacement

# helper for conv dimension calculations
def calc_same_padding(stride=2, width=28, filter_size=5):
    S, W, F = stride, width, filter_size
    P = ((S - 1) * W - S + F) / 2
    padding_height = (stride * (width - 1) + filter_size - width) / 2
    return int(np.round(padding_height))


def calc_output_padding(s=2, p=0, i=2, o=0, k=3):
    output_padding = (o - (i - 1) * s) + 2 * p - k
    return int(np.round(output_padding))


def calc_output_conv(s=2, p=0, i=2, k=3):
    o = (i + 2 * p - k) / s + 1
    if o <= 0:
        raise ValueError(
            "Invalid output size calculated %d<=0. Try different kernel and stride sizes."
            % o
        )
    return int(np.round(o))


def calc_output_convtranspose(s=2, p=0, i=2, output_padding=0, k=3):
    o = (i - 1) * s - 2 * p + k + output_padding
    if o <= 0:
        raise ValueError(
            "Invalid output size calculated %d<=0. Try different kernel and stride sizes."
            % o
        )
    return int(np.round(o))


def convert_tuple_conv2d_params(input_dim, *args):
    """
    Converts the parameters of int into a tuple of equal int. If it is already a tuple, then ignore and return the same.

    Parameters
    ----------
    input_dim : int or tuple
    *args : list of int or list of tuple
        for each item, we convert the int into a tuple of int. If it is already a tuple, then return the same.

    Return
    ------
    input_dim : tuple
    *args : list of tuple

    """
    # convert a single input into tuple
    if isinstance(input_dim, int):
        input_dim = (input_dim, input_dim)

    if len(args) == 0:
        return input_dim
    else:
        temp_args = list(args)
        for i, arg in enumerate(temp_args):
            if isinstance(arg, tuple):
                arg = list(arg)
                temp_args[i] = arg
            for j, arg_inner in enumerate(arg):
                if isinstance(arg_inner, int):
                    arg[j] = [arg_inner, arg_inner]

        return (input_dim, *temp_args)


def calc_conv2dforward_pass(input_dim=28, strides=[], paddings=[], kernels=[]):
    """
    Calculates the dimensions (width and height) of the output from a series of Conv2D layer, given the input dimensions, strides, paddings and kernels
    If only int is supplied for any of the parameters, a tuple of equal dimensions is internally supplied.
    For strides, paddings and kernels, list of int or tuple of int are expected, where each item in the list
    correspond to multiple layers chained together.

    Parameters
    ----------
    input_dim : int or tuple of int
        Input dimensions of the data.

    strides : list of int or list of tuple of int
        Parameters of Conv2D layer.
    paddings : list of int or list of tuple of int
        Parameters of Conv2D layer.
    kernels : list of int or list of tuple of int
        Parameters of Conv2D layer.

    Returns
    -------
    output_dimensions : list of tuples
        Output dimensions for each layer in the given list of parameters
    """
    output_dimensions = []
    # convert a single input into tuple
    input_dim, strides, paddings, kernels = convert_tuple_conv2d_params(
        input_dim, strides, paddings, kernels
    )

    for num_layer, _ in enumerate(kernels):
        if num_layer == 0:
            input_dim = input_dim
        else:
            input_dim = output_dimensions[-1]
        output_dimensions.append(
            (
                calc_output_conv(
                    s=strides[num_layer][0],
                    p=paddings[num_layer][0],
                    i=input_dim[0],
                    k=kernels[num_layer][0],
                ),
                calc_output_conv(
                    s=strides[num_layer][1],
                    p=paddings[num_layer][1],
                    i=input_dim[1],
                    k=kernels[num_layer][1],
                ),
            )
        )
    return output_dimensions


def calc_conv1dforward_pass(input_dim=28, strides=[], paddings=[], kernels=[]):
    """
    Similar to `calc_conv2dforward_pass` but for series of Conv1D layers instead of Conv2D to calculate the output dimensions.
    Hence, different from `calc_conv2dforward_pass`, we expect list of ints for the parameters instead of tuple of size 2

    Parameters
    ----------
    input_dim : int
        Input dimensions of the 1D data.

    strides : list of int
        Parameters of Conv1D layer.
    paddings : list of int
        Parameters of Conv1D layer.
    kernels : list of int
        Parameters of Conv1D layer.

    Returns
    -------
    output_dimensions : list
        Output dimensions for each layer in the given list of parameters
    """
    output_dimensions = []

    for num_layer, _ in enumerate(kernels):
        if num_layer == 0:
            input_dim = input_dim
        else:
            input_dim = output_dimensions[-1]
        output_dimensions.append(
            calc_output_conv(
                s=strides[num_layer],
                p=paddings[num_layer],
                i=input_dim,
                k=kernels[num_layer],
            )
        )
    return output_dimensions


def calc_flatten_conv2d_forward_pass(
    input_dim=28, channels=[], strides=[], paddings=[], kernels=[], flatten=True
):
    """
    Calculates the output dimensions of Conv2D layers, taking into account the number of channels.
    Provides an option to flatten the dimensions which is used when inferring the size of Dense layer that this series of Conv2D layers will be connected to.

    Uses `calc_conv2dforward_pass` for calculating the height and width dimensions.

    Parameters
    ----------
    input_dim : int or tuple of int
        Input dimensions of the data.

    channels : list of int or list of tuple of int
        Parameters of Conv2D layer.
    strides : list of int or list of tuple of int
        Parameters of Conv2D layer.
    paddings : list of int or list of tuple of int
        Parameters of Conv2D layer.
    kernels : list of int or list of tuple of int
        Parameters of Conv2D layer.

    flatten : bool
        Choose whether to flatten the dimensions of width x height x channels or not.
        If True, the output will be the flattened dimensions for every Conv2D layer in the list.
        If False, the output will be the (channels, (width, height))

    Returns
    -------
    dimensions : list of tuples
        Output dimensions for each layer in the given list of parameters. The format depends on `flatten` parameter.

    """
    # convert a single input into tuple
    input_dim, strides, paddings, kernels = convert_tuple_conv2d_params(
        input_dim, strides, paddings, kernels
    )

    results = [
        (input_dim[0] * input_dim[1]) * channels[0]
        if flatten
        else ((channels[0], input_dim[0]))
    ]
    dims = calc_conv2dforward_pass(
        input_dim=input_dim, strides=strides, paddings=paddings, kernels=kernels
    )
    for dim, channel in zip(dims, channels[1:]):
        if flatten:
            results.append((dim[0] * dim[1]) * channel)
        else:
            results.append((channel, dim))
    return results


def calc_flatten_conv1d_forward_pass(
    input_dim=28, channels=[], strides=[], paddings=[], kernels=[], flatten=True
):
    """
    Similar to `calc_flatten_conv2d_forward_pass` but for Conv1D.

    Parameters
    ----------
    input_dim : int
        Input dimensions of the 1D data.

    strides : list of int
        Parameters of Conv1D layer.
    paddings : list of int
        Parameters of Conv1D layer.
    kernels : list of int
        Parameters of Conv1D layer.
    flatten : bool
        Choose whether to flatten the dimensions of data length x channels or not.
        If True, the output will be the flattened dimensions for every Conv1D layer in the list.
        If False, the output will be the (channels, (data_length))

    Returns
    -------
    dimensions : list of tuples
        Output dimensions for each layer in the given list of parameters. The format depends on `flatten` parameter.

    """
    results = [input_dim * channels[0] if flatten else ((channels[0], input_dim))]
    dims = calc_conv1dforward_pass(
        input_dim=input_dim, strides=strides, paddings=paddings, kernels=kernels
    )
    for dim, channel in zip(dims, channels[1:]):
        if flatten:
            results.append(dim * channel)
        else:
            results.append((channel, dim))
    return results


def calc_target_dims(encode_dims=[28, 12, 4], input_dim_init=0):
    target_dims = copy.copy(encode_dims)
    target_dims.reverse()
    target_dims.append(input_dim_init)
    target_dims = target_dims[1:]
    return target_dims


# calculate for decoder forward pass dimension
def calc_conv2dtranspose_pass(
    input_dim=2, output_padding=[], strides=[], paddings=[], kernels=[]
):
    """
    Similar to `calc_conv2dforward_pass` but for Conv2DTranspose layers instead.
    See `calc_conv2dforward_pass` for the description of expected parameters.
    Note that output_padding is unique to ConvTranspose layers only.
    """
    # reverse it
    s_reverse = list(reversed(strides))
    p_reverse = list(reversed(paddings))
    k_reverse = list(reversed(kernels))

    output_dimensions = []
    # convert a single input into tuple
    (
        input_dim,
        s_reverse,
        p_reverse,
        k_reverse,
        output_padding,
    ) = convert_tuple_conv2d_params(
        input_dim, s_reverse, p_reverse, k_reverse, output_padding
    )

    for num_layer, _ in enumerate(kernels):
        if num_layer == 0:
            input_dim = input_dim
        else:
            input_dim = output_dimensions[-1]
        output_dimensions.append(
            (
                calc_output_convtranspose(
                    s=s_reverse[num_layer][0],
                    p=p_reverse[num_layer][0],
                    i=input_dim[0],
                    k=k_reverse[num_layer][0],
                    output_padding=output_padding[num_layer][0],
                ),
                calc_output_convtranspose(
                    s=s_reverse[num_layer][1],
                    p=p_reverse[num_layer][1],
                    i=input_dim[1],
                    k=k_reverse[num_layer][1],
                    output_padding=output_padding[num_layer][1],
                ),
            )
        )
    return output_dimensions


def calc_conv1dtranspose_pass(
    input_dim=2, output_padding=[], strides=[], paddings=[], kernels=[]
):
    """
    Similar to `calc_conv1dforward_pass` but for Conv1DTranspose layers instead.
    See `calc_conv1dforward_pass` for the description of expected parameters.
    Note that output_padding is unique to ConvTranspose layers only.
    """
    # reverse it
    s_reverse = list(reversed(strides))
    p_reverse = list(reversed(paddings))
    k_reverse = list(reversed(kernels))

    output_dimensions = []

    for num_layer, _ in enumerate(kernels):
        if num_layer == 0:
            input_dim = input_dim
        else:
            input_dim = output_dimensions[-1]
        output_dimensions.append(
            calc_output_convtranspose(
                s=s_reverse[num_layer],
                p=p_reverse[num_layer],
                i=input_dim,
                k=k_reverse[num_layer],
                output_padding=output_padding[num_layer],
            )
        )
    return output_dimensions


def calc_flatten_conv1dtranspose_forward_pass(
    input_dim=28,
    channels=[],
    output_padding=[],
    strides=[],
    paddings=[],
    kernels=[],
    flatten=True,
):
    results = [(input_dim) * channels[0] if flatten else ((channels[0], input_dim))]
    dims = calc_conv2dtranspose_pass(
        input_dim=input_dim,
        output_padding=output_padding,
        strides=strides,
        paddings=paddings,
        kernels=kernels,
    )
    for dim, channel in zip(dims, channels[1:]):
        if flatten:
            results.append(dim * channel)
        else:
            results.append((channel, dim))
    return results


def calc_flatten_conv2dtranspose_forward_pass(
    input_dim=28,
    channels=[],
    output_padding=[],
    strides=[],
    paddings=[],
    kernels=[],
    flatten=True,
):
    results = [
        (input_dim ** 2) * channels[0] if flatten else ((channels[0], input_dim))
    ]
    dims = calc_conv2dtranspose_pass(
        input_dim=input_dim,
        output_padding=output_padding,
        strides=strides,
        paddings=paddings,
        kernels=kernels,
    )
    for dim, channel in zip(dims, channels[1:]):
        if flatten:
            results.append((dim ** 2) * channel)
        else:
            results.append((channel, dim))
    return results


# condition to stop iteration
def reached_target_dims(d_temp, target_dims):
    if np.sum(d_temp) - np.sum(target_dims) != 0:
        return False
    return True


def get_updated_output_padding(output_padding, current_dims, target_dims, kernels):
    # sequential
    temp_output_padding = np.array(output_padding)
    current_dims = np.array(current_dims)
    target_dims = np.array(target_dims)
    kernels = np.array(kernels)

    for num_layer, _ in enumerate(kernels):
        # required_output_padding = calc_output_padding(s=s_reverse[num_layer],p=p_reverse[num_layer],i=decode_input_dim,k=k_reverse[num_layer],o=target_dims[num_layer])
        required_output_padding = target_dims[num_layer] - current_dims[num_layer]
        if required_output_padding.any():
            # fails the test, update the current_output_padding to that of the required one
            temp_output_padding[num_layer] = (
                temp_output_padding[num_layer] + required_output_padding
            )
            return temp_output_padding

    current_dims = list(current_dims)
    target_dims = list(target_dims)
    kernels = list(kernels)

    return list(temp_output_padding)


def calc_required_output_padding(
    input_dim_init=28,
    kernels=[5, 5, 2],
    strides=[2, 2, 2],
    paddings=[0, 0, 0],
    verbose=True,
    conv_dim=2,
):
    if conv_dim == 2:
        encode_dims = calc_conv2dforward_pass(
            input_dim=input_dim_init,
            strides=strides,
            paddings=paddings,
            kernels=kernels,
        )
        if isinstance(input_dim_init, int):
            target_dims = calc_target_dims(
                encode_dims, input_dim_init=[input_dim_init] * 2
            )
        else:
            target_dims = calc_target_dims(encode_dims, input_dim_init=input_dim_init)

    else:
        encode_dims = calc_conv1dforward_pass(
            input_dim=input_dim_init,
            strides=strides,
            paddings=paddings,
            kernels=kernels,
        )
        target_dims = calc_target_dims(encode_dims, input_dim_init=input_dim_init)

    output_padding_init = [0] * len(kernels)

    if conv_dim == 2:
        decode_dims_init = calc_conv2dtranspose_pass(
            input_dim=encode_dims[-1],
            output_padding=output_padding_init,
            strides=strides,
            paddings=paddings,
            kernels=kernels,
        )
    else:
        decode_dims_init = calc_conv1dtranspose_pass(
            input_dim=encode_dims[-1],
            output_padding=output_padding_init,
            strides=strides,
            paddings=paddings,
            kernels=kernels,
        )
    # iterate through
    max_iterations = 100
    current_i = 0
    decode_dims_i = copy.copy(decode_dims_init)
    output_padding_new = copy.copy(output_padding_init)
    if verbose:
        print(
            "REQ OUTPUT PADDING, DEC-DIM, TARG-DIM :\n"
            + str([output_padding_init, decode_dims_i, target_dims])
        )

    while (
        reached_target_dims(decode_dims_i, target_dims) == False
        and current_i < max_iterations
    ):
        output_padding_new = get_updated_output_padding(
            output_padding_new, decode_dims_i, target_dims, kernels
        )
        if conv_dim == 2:
            decode_dims_i = calc_conv2dtranspose_pass(
                input_dim=encode_dims[-1],
                output_padding=output_padding_new,
                strides=strides,
                paddings=paddings,
                kernels=kernels,
            )
        else:
            decode_dims_i = calc_conv1dtranspose_pass(
                input_dim=encode_dims[-1],
                output_padding=output_padding_new,
                strides=strides,
                paddings=paddings,
                kernels=kernels,
            )

        output_padding_new = np.array(output_padding_new)
        if np.any(output_padding_new == -1):
            return -1

        current_i += 1

    output_padding_new = list(output_padding_new)
    for i, output_padding_inner in enumerate(output_padding_new):
        if isinstance(output_padding_inner, np.ndarray):
            output_padding_new[i] = list(output_padding_inner)

    if verbose:
        print([output_padding_new, decode_dims_i, target_dims])

    return output_padding_new


def calc_required_padding(
    input_dim_init=28, kernels=[5, 5, 2], strides=[2, 2, 2], verbose=True, conv_dim=2
):
    """
    Calculates the required paddings and output paddings for Conv and ConvTranspose to achieve a symmetrical input-output size for use of autoencoder.

    This will be used in the `ConvLayers` class.

    """
    init_paddings = [0] * len(kernels)
    output_padding = calc_required_output_padding(
        input_dim_init=input_dim_init,
        kernels=kernels,
        strides=strides,
        paddings=init_paddings,
        verbose=verbose,
        conv_dim=conv_dim,
    )

    if isinstance(output_padding, int) == True:
        possible_paddings = list(combinations_with_replacement([0, 1, 2], len(kernels)))
        for id_, paddings in enumerate(possible_paddings):
            if verbose:
                print(paddings)
            output_padding = calc_required_output_padding(
                input_dim_init=input_dim_init,
                kernels=kernels,
                strides=strides,
                paddings=paddings,
                verbose=verbose,
                conv_dim=conv_dim,
            )
            if isinstance(output_padding, int) == False:
                return list(paddings), output_padding
    return init_paddings, output_padding


"""
#EXAMPLES:
#Calculate required padding for (de)convolutional layers to achieve symmetrical filter/strides/kernel sizes

#Specify num channels, kernels, and strides
conv_architecture=[1,32,64,128]
conv_kernel=[4,4,4]
conv_stride=[2,2,2]

#calculate required conv paddings
conv_padding, output_padding = calc_required_padding(input_dim_init=28, kernels=conv_kernel, strides=conv_stride,verbose=True)

#supply the parameters to instantiate ConvLayers and its transpose (with upsampling=True)
conv_layer = ConvLayers(conv_architecture=conv_architecture, conv_kernel=conv_kernel, conv_stride=conv_stride)
deconv_layer = ConvLayers(upsampling=True,conv_architecture=conv_architecture, conv_kernel=conv_kernel, conv_stride=conv_stride)

"""
