import torch
import numpy as np
from ..util.misc import parse_activation


def compute_entropy(vector_s):
    return (np.log(vector_s) * vector_s).sum() * -1


def parse_layer_size(input_size, layer_string):
    """
    Parse the layer size and operator to get the magnitude

    Parameters
    ----------
    input_size : int
        Input size of current layer
    layer_string : str or int
        If string, two operators are accepted : "d2" or "x2" where d means division and x means multiplication.
        The number represents the magnitude of operator to be applied on the `input_size`, in the example it is 2.
        If int is passed, the instantiated layer output size will be the the exact int itself and `input_size` is ignored.
    """
    if isinstance(layer_string, str):
        operator = layer_string[0]
        if len(layer_string) > 1:
            magnitude = float(layer_string[1:])
        else:
            magnitude = operator

        if operator.lower() == "d":
            output_size = int(input_size / magnitude)
        elif operator.lower() == "x":
            output_size = int(input_size * magnitude)
        else:
            output_size = int(layer_string)
    else:
        output_size = int(layer_string)
    return output_size


def append_activation(layers, activation):
    activation_layer = parse_activation(activation)
    if activation_layer is not None:
        layers.append(activation_layer)
    return layers


def parse_architecture_string(
    input_size,
    output_size,
    architecture,
    layer_type=torch.nn.Linear,
    activation="relu",
    last_activation="none",
    bias=False,
    norm=True,
    last_norm=True,
):
    """
    Parses a list of string representing the hidden layer sizes with option for operator to get the magnitude
    For example: `input_size` of 10, `output_size` of 1, and `architecture` of ["d2","x3"]
    will create a Neural net with 10-5-15-1 nodes, with two hidden layers.

    Parameters
    ----------
    input_size : int
        Input size of the Neural network

    output_size : int
        Output size of the Neural network

    architecture : list of str or int
        List of architecture string representing the hidden nodes.
        Each string in the list represents an operator and magnitude to be applied.
        For exact number of nodes, pass list of int instead

    layer_type : torch.nn.Linear
        Type of layer to be instantiated for each architecture string.
    """

    layers = []
    if len(architecture) == 0:
        layers.append(layer_type(input_size, output_size, bias=bias))
        # last activation layer
        layers = append_activation(layers, last_activation)

    else:
        for layer_index, layer_string in enumerate(architecture):
            num_nodes = parse_layer_size(input_size, layer_string)
            if layer_index == 0:
                first_layer = layer_type(input_size, num_nodes, bias=bias)
                layers.append(first_layer)
                # if norm:
                #     layers.append(torch.nn.BatchNorm1d(num_features=num_nodes))

                layers = append_activation(layers, activation)

                # special case if there's only a single hidden layer
                if len(architecture) == 1:
                    last_layer = layer_type(
                        num_nodes,
                        output_size,
                        bias=bias,
                    )
                    layers.append(last_layer)

                    # last activation layer
                    layers = append_activation(layers, last_activation)

            elif layer_index == (len(architecture) - 1):
                new_layer = layer_type(
                    parse_layer_size(input_size, architecture[layer_index - 1]),
                    num_nodes,
                    bias=bias,
                )
                last_layer = layer_type(num_nodes, output_size, bias=bias)
                layers.append(new_layer)
                # if norm:
                #     layers.append(torch.nn.BatchNorm1d(num_features=num_nodes))
                layers = append_activation(layers, activation)
                layers.append(last_layer)
                # if last_norm:
                #     layers.append(torch.nn.BatchNorm1d(num_features=output_size))
                layers = append_activation(layers, last_activation)
            else:
                new_layer = layer_type(
                    parse_layer_size(input_size, architecture[layer_index - 1]),
                    num_nodes,
                    bias=bias,
                )
                layers.append(new_layer)
                # if norm:
                #     layers.append(torch.nn.BatchNorm1d(num_features=num_nodes))
                layers = append_activation(layers, activation)

    return layers


def create_dense_block(
    num_nodes,
    output_size,
    bias=False,
    norm=True,
    activation="none",
    layer_type=torch.nn.Linear,
):
    layers = []
    new_layer = layer_type(
        num_nodes,
        output_size,
        bias=bias,
    )
    layers.append(new_layer)

    # add batch norm
    if norm:
        layers.append(torch.nn.BatchNorm1d(num_features=output_size))

    # last activation layer
    layers = append_activation(layers, activation)

    return torch.nn.Sequential(*layers)


def parse_architecture_string_v2(
    input_size,
    output_size,
    architecture,
    layer_type=torch.nn.Linear,
    activation="relu",
    last_activation="none",
    bias=False,
    norm=True,
    last_norm=True,
):
    """
    Parses a list of string representing the hidden layer sizes with option for operator to get the magnitude
    For example: `input_size` of 10, `output_size` of 1, and `architecture` of ["d2","x3"]
    will create a Neural net with 10-5-15-1 nodes, with two hidden layers.

    Parameters
    ----------
    input_size : int
        Input size of the Neural network

    output_size : int
        Output size of the Neural network

    architecture : list of str or int
        List of architecture string representing the hidden nodes.
        Each string in the list represents an operator and magnitude to be applied.
        For exact number of nodes, pass list of int instead

    layer_type : torch.nn.Linear
        Type of layer to be instantiated for each architecture string.
    """

    blocks = []
    if len(architecture) == 0:
        block = create_dense_block(
            num_nodes=input_size,
            output_size=output_size,
            bias=bias,
            norm=norm,
            activation=last_activation,
        )
        blocks.append(block)
    else:
        for layer_index, layer_string in enumerate(architecture):
            num_nodes = parse_layer_size(input_size, layer_string)
            if layer_index == 0:
                block = create_dense_block(
                    num_nodes=input_size,
                    output_size=num_nodes,
                    bias=bias,
                    norm=norm,
                    activation=activation,
                )
                blocks.append(block)

                # special case if there's only a single hidden layer
                if len(architecture) == 1:
                    block = create_dense_block(
                        num_nodes=num_nodes,
                        output_size=output_size,
                        bias=bias,
                        norm=last_norm,
                        activation=last_activation,
                    )
                    blocks.append(block)

            elif layer_index == (len(architecture) - 1):
                # new_layer = layer_type(
                #     parse_layer_size(input_size, architecture[layer_index - 1]),
                #     num_nodes,
                #     bias=bias,
                # )
                # last_layer = layer_type(num_nodes, output_size, bias=bias)
                # layers.append(new_layer)
                # if norm:
                #     layers.append(torch.nn.BatchNorm1d(num_features=num_nodes))
                # layers = append_activation(layers, activation)
                # layers.append(last_layer)
                # if last_norm:
                #     layers.append(torch.nn.BatchNorm1d(num_features=output_size))
                # layers = append_activation(layers, last_activation)

                block = create_dense_block(
                    num_nodes=input_size,
                    output_size=num_nodes,
                    bias=bias,
                    norm=norm,
                    activation=activation,
                )
                blocks.append(block)
                block = create_dense_block(
                    num_nodes=num_nodes,
                    output_size=parse_layer_size(
                        input_size, architecture[layer_index - 1]
                    ),
                    bias=bias,
                    norm=last_norm,
                    activation=last_activation,
                )
                blocks.append(block)

            else:
                block = create_dense_block(
                    num_nodes=parse_layer_size(
                        input_size, architecture[layer_index - 1]
                    ),
                    output_size=num_nodes,
                    bias=bias,
                    norm=norm,
                    activation=activation,
                )
                blocks.append(block)
    return blocks


def convert_chol_tril(chol_tril):
    """
    Converts Cholesky lower triangle of inverse matrix, to that of covariance matrix
    The base `log_chol_tril` is expected to be a 2D matrix with upper triangle zeros.
    Checks the dimension of data, if 3D, assume it to be batched, otherwise assume as non-batched
    """
    # batch mode
    if len(chol_tril.shape) >= 3:
        chol_tril = torch.from_numpy(chol_tril).float()
        reconstructed_precision_mat = torch.matmul(
            chol_tril, torch.transpose(chol_tril, 2, 1)
        )
        covariance_matrix = [
            torch.inverse(reconstructed_precision_mat[i])
            for i in range(chol_tril.shape[0])
        ]
        covariance_matrix = torch.stack(covariance_matrix).detach().cpu().numpy()
        return covariance_matrix

    else:
        reconstructed_precision_mat = np.matmul(chol_tril, np.transpose(chol_tril))
        covariance_matrix = np.linalg.inv(reconstructed_precision_mat)

    return covariance_matrix
