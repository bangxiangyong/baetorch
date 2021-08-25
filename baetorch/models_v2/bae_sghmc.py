import copy

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn import Parameter

# from .sparse_ae import SparseAutoencoderModule
from .base_autoencoder import BAE_BaseClass
from ..models_v2.base_layer import (
    Reshape,
    Flatten,
    TwinOutputModule,
    get_conv_latent_shapes,
    create_linear_chain,
    create_conv_chain,
)
import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.autograd import Variable
from torch.nn import Parameter
from tqdm import tqdm

from ..models.base_layer import (
    ConvLayers,
    Conv2DLayers,
    Conv1DLayers,
    DenseLayers,
    Reshape,
    Flatten,
    flatten_torch,
    flatten_np,
)
from ..models.cholesky_layer import CholLayer
from ..util.distributions import CB_Distribution, TruncatedGaussian
from ..util.minmax import TorchMinMaxScaler
from ..util.misc import create_dir
from ..util.sghmc import SGHMC
from ..util.truncated_gaussian import TruncatedNormal


class BAE_SGHMC(BAE_BaseClass):
    burn_stage = True
    base_learning_rate = 0.1
    # base_learning_rate = 1.0
    continous_fit = False
    sghmc_params = []

    def get_optimisers(self, autoencoder):
        optimiser_list = self.get_optimisers_list(autoencoder)
        if self.burn_stage:
            return torch.optim.Adam(optimiser_list, lr=self.learning_rate)
        else:
            return SGHMC(
                optimiser_list,
                lr=self.base_learning_rate,
                num_burn_in_steps=10,
                scale_grad=0.0001,
            )

    def init_fit(self):
        # init optimisers and scheduler for fitting model
        if not self.continous_fit:
            self.set_optimisers()
        if self.scheduler_enabled and self.burn_stage:
            self.init_scheduler()

    def fit(
        self,
        x,
        y=None,
        burn_epoch=100,
        sghmc_epoch=200,
        clear_sghmc_params=True,
        **fit_kwargs
    ):
        """
        There are two phases - one of burn-in and second of sampling .
        During burn-in , Adam is used as usual training of a deterministic AE.
        Then, sampling runs from

        """
        if clear_sghmc_params:
            self.sghmc_params = []

        if burn_epoch > 0:
            self.burn_stage = True
            super(BAE_SGHMC, self).fit(x, y=y, num_epochs=burn_epoch, **fit_kwargs)
            self.save_sghmc_parameters()

        if sghmc_epoch > 0:
            self.burn_stage = False
            self.scheduler_enabled = True

            # save every N mini batch to fulfill criteria of required samples
            if isinstance(x, torch.utils.data.dataloader.DataLoader):
                self.save_every = (len(x) * sghmc_epoch) // self.num_samples
            else:
                self.save_every = sghmc_epoch // self.num_samples

            self.save_every_counter = 0  # counter
            for i in range(sghmc_epoch):
                self.continous_fit = False if i == 0 else True

                super(BAE_SGHMC, self).fit(x, y=y, num_epochs=1, **fit_kwargs)
            self.continous_fit = False
            self.save_every_counter = 0

    def step_scheduler(self):
        """
        Override the scheduler to behave as usual during burn stage.
        But, on sampling stage, we use the scheduler func to collect samples.
        """
        if self.burn_stage:
            for scheduler in self.scheduler:
                scheduler.step()
        else:
            self.save_every_counter += 1
            if self.save_every_counter % self.save_every == 0:
                # collect sample if limit has not reached yet
                if len(self.sghmc_params) < self.num_samples:
                    self.save_sghmc_parameters()

    def save_sghmc_parameters(self):
        self.sghmc_params.append(copy.deepcopy(self.autoencoder.state_dict()))

    def predict(
        self,
        x,
        y=None,
        select_keys=["y_mu", "y_sigma", "se", "bce", "nll"],
        *args,
        **params
    ):

        # get individual forward predictions
        predictions = []
        for sghmc_param in self.sghmc_params:
            self.autoencoder.load_state_dict(state_dict=sghmc_param)
            prediction = self.predict_(
                x=x,
                y=y,
                select_keys=select_keys,
                autoencoder_=self.autoencoder,
                *args,
                **params
            )
            predictions.append(prediction)

        # stack them
        stacked_predictions = self.concat_predictions(predictions)

        return stacked_predictions
