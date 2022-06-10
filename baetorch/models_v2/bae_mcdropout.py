import torch

from baetorch.baetorch.models_v2.base_autoencoder import BAE_BaseClass


class BAE_MCDropout(BAE_BaseClass):
    def __init__(
        self, num_train_samples=1, num_test_samples=100, dropout_rate=0.1, **params
    ):
        # store variables specific to mc_dropout
        self.dropout_rate = dropout_rate
        self.num_train_samples = num_train_samples

        # enforce dropout layer
        for param in params["chain_params"]:
            param.update(
                {
                    "dropout": dropout_rate,
                    "dropout_n": num_test_samples
                    if "num_samples" not in params
                    else params["num_samples"],
                }
            )
        super(BAE_MCDropout, self).__init__(model_type="stochastic", **params)

    def log_prior_loss(self, model):
        prior_loss = super(BAE_MCDropout, self).log_prior_loss(model)
        prior_loss *= 1.0 - self.dropout_rate
        return prior_loss

    def criterion(self, autoencoder, x, y=None):
        """
        `autoencoder` here is a list of autoencoder
        We sum the losses and backpropagate them at one go
        """
        stacked_criterion = torch.stack(
            [
                super(BAE_MCDropout, self).criterion(self.autoencoder, x, y=y)
                for i in range(self.num_train_samples)
            ]
        )
        return stacked_criterion.mean()
