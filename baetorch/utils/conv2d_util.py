import numpy as np
import copy
from itertools import combinations_with_replacement

#helper for conv dimension calculations
def calc_same_padding(stride=2,width=28,filter_size=5):
    S,W,F =stride,width,filter_size
    P = ((S-1)*W-S+F)/2
    # P = ((stride-1)*width-stride+filter_size)/2
    padding_height = (stride * (width - 1) + filter_size - width) / 2
    return int(np.round(padding_height))

def calc_output_padding(s=2,p=0,i=2,o = 0, k=3):
    output_padding = (o -(i -1)*s)+2*p-k
    return int(np.round(output_padding))

def calc_output_conv2d(s=2,p=0,i=2, k=3):
    o = (i + 2*p - k)/s + 1
    return int(np.round(o))

def calc_output_conv2dtranspose(s=2,p=0,i=2,output_padding = 0, k=3):
    o = (i -1)*s - 2*p + k + output_padding
    return int(np.round(o))

def calc_conv2dforward_pass(input_dim=28, strides=[], paddings=[], kernels=[]):
    e_ = []
    for num_layer, _ in enumerate(kernels):
        if num_layer == 0:
            input_dim = input_dim
        else:
            input_dim = e_[-1]
        e_.append(calc_output_conv2d(s=strides[num_layer],p=paddings[num_layer],i=input_dim,k=kernels[num_layer]))
    return e_

def calc_flatten_conv2d_forward_pass(input_dim=28, channels=[], strides=[], paddings=[], kernels=[], flatten=True):
    results=[(input_dim**2)*channels[0] if flatten else ((channels[0],input_dim))]
    dims = calc_conv2dforward_pass(input_dim=input_dim,strides=strides,paddings=paddings,kernels=kernels)
    for dim, channel in zip(dims,channels[1:]):
        if flatten:
            results.append((dim**2)*channel)
        else:
            results.append((channel, dim))
    return results

def calc_target_dims(encode_dims=[28, 12, 4], input_dim_init=0):
    target_dims = copy.copy(encode_dims)
    target_dims.reverse()
    target_dims.append(input_dim_init)
    target_dims = target_dims[1:]
    return target_dims

#calculate for decoder forward pass dimension
def calc_conv2dtranspose_pass(init_e_dim=2,output_padding=[],strides=[],paddings=[], kernels=[]):
    #reverse it
    s_reverse = list(reversed(strides))
    p_reverse = list(reversed(paddings))
    k_reverse = list(reversed(kernels))
    d_dims = []
    for num_layer, _ in enumerate(kernels):
        if num_layer == 0:
            decode_input_dim = init_e_dim
        else:
            decode_input_dim = d_dims[-1]
        d_dims.append(calc_output_conv2dtranspose(s=s_reverse[num_layer],p=p_reverse[num_layer],i=decode_input_dim,k=k_reverse[num_layer],output_padding=output_padding[num_layer]))
    return d_dims

def calc_flatten_conv2dtranspose_forward_pass(input_dim=28, channels=[], output_padding=[], strides=[], paddings=[], kernels=[], flatten=True):
    results=[(input_dim**2)*channels[0] if flatten else ((channels[0],input_dim))]
    dims = calc_conv2dtranspose_pass(init_e_dim=input_dim,output_padding=output_padding,strides=strides,paddings=paddings,kernels=kernels)
    for dim, channel in zip(dims,channels[1:]):
        if flatten:
            results.append((dim**2)*channel)
        else:
            results.append((channel, dim))
    return results

#condition to stop iteration
def reached_target_dims(d_temp, target_dims):
    if np.sum(d_temp)-np.sum(target_dims) != 0:
        return False
    return True

def get_updated_output_padding(output_padding, current_dims, target_dims, kernels):
    #sequential
    temp_output_padding = copy.copy(output_padding)
    for num_layer, _ in enumerate(kernels):
        # required_output_padding = calc_output_padding(s=s_reverse[num_layer],p=p_reverse[num_layer],i=decode_input_dim,k=k_reverse[num_layer],o=target_dims[num_layer])
        required_output_padding =target_dims[num_layer]-current_dims[num_layer]
        if required_output_padding != 0:
            #fails the test, update the current_output_padding to that of the required one
            temp_output_padding[num_layer] = temp_output_padding[num_layer]+required_output_padding
            return temp_output_padding
    return temp_output_padding

def calc_required_output_padding(input_dim_init=28, kernels=[5,5,2],strides=[2,2,2], paddings=[0,0,0],verbose=True):
    encode_dims = calc_conv2dforward_pass(input_dim=input_dim_init, strides=strides, paddings=paddings, kernels=kernels)
    target_dims = calc_target_dims(encode_dims, input_dim_init=input_dim_init)

    output_padding_init = [0] * len(kernels)
    decode_dims_init = calc_conv2dtranspose_pass(init_e_dim=encode_dims[-1], output_padding=output_padding_init, strides=strides, paddings=paddings, kernels=kernels)

    #iterate through
    max_iterations = 100
    current_i =0
    decode_dims_i = copy.copy(decode_dims_init)
    output_padding_new = copy.copy(output_padding_init)
    if verbose:
        print([output_padding_init,decode_dims_i,target_dims])
    while (reached_target_dims(decode_dims_i, target_dims) == False and current_i < max_iterations):
        output_padding_new = get_updated_output_padding(output_padding_new, decode_dims_i, target_dims, kernels)
        decode_dims_i = calc_conv2dtranspose_pass(init_e_dim=encode_dims[-1], output_padding=output_padding_new, strides=strides, paddings=paddings, kernels=kernels)
        current_i+=1

        if verbose:
            print([output_padding_new,decode_dims_i,target_dims])
        if -1 in output_padding_new:
            return -1
    return output_padding_new

def calc_required_padding(input_dim_init=28, kernels=[5, 5, 2], strides=[2, 2, 2], verbose=True):
    init_paddings = [0]*len(kernels)
    output_padding = calc_required_output_padding(input_dim_init=input_dim_init, kernels=kernels,strides=strides, paddings=init_paddings,verbose=verbose)
    if isinstance(output_padding,int) == True:
        possible_paddings = list(combinations_with_replacement([0,1,2], len(kernels)))
        for id_, paddings in enumerate(possible_paddings):
            if verbose:
                print(paddings)
            output_padding = calc_required_output_padding(input_dim_init=input_dim_init, kernels=kernels,strides=strides, paddings=paddings,verbose=verbose)
            if isinstance(output_padding,int) == False:
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
