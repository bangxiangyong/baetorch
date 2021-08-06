import torch
from ..models_v2.base_autoencoder import BAE_BaseClass
import numpy as np

# Ensemble
class BAE_Ensemble(BAE_BaseClass):
    def __init__(self, *args, num_samples=5, model_name="BAE_Ensemble", **kwargs):
        super(BAE_Ensemble, self).__init__(
            *args,
            num_samples=num_samples,
            model_name=model_name,
            model_type="list",
            **kwargs
        )

        for autoencoder_ in self.autoencoder:
            autoencoder_.reset_parameters()

    def init_autoencoder_module(self, AE_Module, **params):
        """
        Override this if required to init different AE Module.
        """

        return [AE_Module(**params) for sample in range(self.num_samples)]

    # def init_autoencoder(self, autoencoder):
    #     """
    #     Initialise an ensemble of autoencoders
    #     """
    #     self.autoencoder = [
    #         super(BAE_Ensemble, self).init_autoencoder(autoencoder)
    #         for sample in range(self.num_samples)
    #     ]
    #     for autoencoder_ in self.autoencoder:
    #         autoencoder_.reset_parameters()
    #     return self.autoencoder

    def criterion(self, autoencoder, x, y=None):
        """
        `autoencoder` here is a list of autoencoder
        We sum the losses and backpropagate them at one go
        """
        stacked_criterion = torch.stack(
            [
                super(BAE_Ensemble, self).criterion(autoencoder=autoencoder, x=x, y=y)
                for autoencoder in self.autoencoder
            ]
        )
        return stacked_criterion.sum()

    # def predict(
    #     self, x, y=None, select_keys=["y_mu", "y_sigma", "se", "nll"], *args, **params
    # ):
    #
    #     # get individual forward predictions
    #     predictions = [
    #         super(BAE_Ensemble, self).predict(
    #             x=x,
    #             y=y,
    #             select_keys=select_keys,
    #             autoencoder_=autoencoder,
    #             *args,
    #             **params
    #         )
    #         for autoencoder in self.autoencoder
    #     ]
    #
    #     # stack them
    #     stacked_predictions = {
    #         key: np.concatenate(
    #             [np.expand_dims(pred_[key], 0) for pred_ in predictions]
    #         ).mean(0)
    #         for key in select_keys
    #     }
    #
    #     return stacked_predictions
