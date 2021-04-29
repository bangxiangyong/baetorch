import torch
import numpy as np

from ..models.base_layer import DenseLayers


class CholLinear(torch.nn.Module):
    """
    Assume the raw output of the dense layer is lower Chol triangular of a precision matrix.
    Note the diagonal is a logged version which will be exponentiated and upper will be zerorised to recover full L.
    Ultimately, it returns L, and log diagonal to calculate neg loglik later.
    """
    def __init__(self, input_size, output_size):
        super(CholLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lower_non_diag_size = ((output_size**2)-output_size)/2
        self.chol_tril_layer = torch.nn.Linear(input_size, output_size+int(self.lower_non_diag_size))

    def batch_matrix_diag(self,mat):
        if mat.dim() == 2:
            mat_diag= mat.as_strided((mat.shape[0],mat.shape[1]), [mat.stride(0), mat.size(2) + 1])
        else:
            mat_diag= torch.diag(mat)
        return mat_diag

    def get_chol_tril_logdiag(self,dense_layer_output, diagonal_size):
        if dense_layer_output.dim() == 2:
            log_diag = dense_layer_output[:,0:diagonal_size]
            batch_size = dense_layer_output.shape[0]
            #now to recover L
            chol_tril = torch.autograd.Variable(torch.zeros((batch_size,diagonal_size,diagonal_size)))
            lii,ljj = np.tril_indices(chol_tril.size(-2), k=-1)
            dii,djj = np.diag_indices(chol_tril.size(-2))
            chol_tril[...,lii,ljj] = torch.exp(dense_layer_output[:,diagonal_size:])
            chol_tril[...,dii,djj] = torch.exp(log_diag)
        else:
            log_diag = dense_layer_output[0:diagonal_size]
            #now to recover L
            chol_tril = torch.autograd.Variable(torch.zeros((diagonal_size,diagonal_size)))
            tril_index = np.tril_indices(diagonal_size)
            diag_index = np.diag_indices(diagonal_size)
            chol_tril[tril_index] = dense_layer_output
            chol_tril[diag_index] = torch.exp(log_diag)
        return chol_tril, log_diag

    def forward(self, x):
        chol_lower_tri_torch = self.chol_tril_layer(x)
        diagonal_size = self.output_size
        chol_trils, log_diags = self.get_chol_tril_logdiag(chol_lower_tri_torch,diagonal_size)
        return chol_trils, log_diags

class CholLayer(torch.nn.Module):
    """
    This combines Dense layers and Chol Linear at the end to form the decoder sigma with full covariance.
    """
    def __init__(self, architecture=[], activation='tanh', last_activation='none', input_size=10, output_size=10, alpha=1.0):
        super(CholLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha

        if len(architecture) != 0:
            if len(architecture) == 1:
                self.dense_decoder = DenseLayers(architecture=[],
                                                           output_size=architecture[-1],
                                                           input_size=self.input_size,
                                                           activation=activation,
                                                           last_activation=last_activation)
            else:
                self.dense_decoder = DenseLayers(architecture=architecture[:-1],
                                                           output_size=architecture[-1],
                                                           input_size=self.input_size,
                                                           activation=activation,
                                                           last_activation=last_activation)
            self.chol_layer = CholLinear(architecture[-1],output_size)
        else:
            self.dense_decoder = None
            self.chol_layer = CholLinear(input_size,output_size)

    def forward(self, x):
        if self.dense_decoder is not None:
            x = self.dense_decoder(x)
        chol_trils, log_diags = self.chol_layer(x)
        return (chol_trils, log_diags)
