import torch
import copy
from torch.nn import Parameter
import torch.nn.functional as F
import bnn.bnn_utils as bnn_utils
from torchvision import datasets, transforms
import numpy as np
from bnn.develop.bayesian_autoencoders.conv2d_util import calc_required_padding, calc_flatten_conv2d_forward_pass
from bnn.develop.bayesian_autoencoders.conv2d_util import calc_required_padding, calc_flatten_conv2d_forward_pass
from bnn.develop.bayesian_autoencoders.base_autoencoder import *
from torch.autograd import Variable
from torch.distributions import Normal, Uniform, Categorical,MultivariateNormal
import copy

#Ensemble
class BAE_Ensemble(BAE_BaseClass):
    def __init__(self,*args, model_name="BAE_Ensemble", **kwargs):
        super(BAE_Ensemble, self).__init__(*args, model_name=model_name, model_type="list", **kwargs)

    def init_autoencoder(self, autoencoder):
        """
        Initialise an ensemble of autoencoders
        """
        self.autoencoder = [super(BAE_Ensemble, self).init_autoencoder(autoencoder) for sample in range(self.num_samples)]
        for autoencoder_ in self.autoencoder:
            autoencoder_.reset_parameters()
        return self.autoencoder

    def criterion(self, autoencoder, x,y=None, mode="mu"):
        """
        `autoencoder` here is a list of autoencoder
        We sum the losses and backpropagate them at one go
        """
        stacked_criterion = torch.stack([super(BAE_Ensemble, self).criterion(autoencoder=model, x=x,y=y, mode=mode) for model in autoencoder])
        return stacked_criterion.sum()
