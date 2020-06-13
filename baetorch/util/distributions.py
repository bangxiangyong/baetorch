import torch
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
import torch.nn.functional as F
import numpy as np

class CB_Distribution():
    def __init__(self, probs=0):
        self.probs = probs
        self._lims = (0.499, 0.501)

    def sumlogC(self, x , eps = 1e-5):
        '''
        Reference code : https://github.com/Robert-Alonso/Continuous-Bernoulli-VAE

        Numerically stable implementation of
        sum of logarithm of Continous Bernoulli
        constant C, using Taylor 2nd degree approximation

        Parameter
        ----------
        x : Tensor of dimensions (batch_size, dim)
            x takes values in (0,1)
        '''
        x = torch.clamp(x, eps, 1.-eps)
        mask = torch.abs(x - 0.5).ge(eps)
        far = torch.masked_select(x, mask)
        close = torch.masked_select(x, ~mask)
        far_values =  torch.log( (torch.log(1. - far) - torch.log(far)).div(1. - 2. * far) )
        close_values = torch.log(torch.tensor((2.))) + torch.log(1. + torch.pow( 1. - 2. * close, 2)/3. )
        return far_values.sum() + close_values.sum()

    def log_bernoulli_loss(self, y_pred_mu, y_true):
        bce = -(y_true*torch.log(y_pred_mu) + (1-y_true)*torch.log(1-y_pred_mu))
        return bce

    def log_cbernoulli_loss_torch(self, y_pred_mu, y_true, mode="robert"):
        if mode == "robert":
            log_p = F.binary_cross_entropy(y_pred_mu,y_true,reduction="sum") -self.sumlogC(y_pred_mu)
            log_p /= y_pred_mu.numel()
        else:
            cb = ContinuousBernoulli(probs=y_pred_mu)
            log_p = -cb.log_prob(y_true)
        return log_p

    def log_cbernoulli_loss_np(self, y_pred_mu, y_true):
        cb = ContinuousBernoulli(probs=y_pred_mu)
        nlog_p = -cb.log_prob(y_true)
        nlog_p = nlog_p.detach().cpu().numpy()
        return nlog_p

    def log_cbernoulli_norm_const_np(self, y_pred_mu, y_true, l_lim=0.49, u_lim=0.51):
        cb = ContinuousBernoulli(probs=y_pred_mu)
        log_p = -cb._cont_bern_log_norm(y_true)
        return np.where(np.logical_or(np.less(y_pred_mu, l_lim), np.greater(y_pred_mu, u_lim)), log_norm, taylor)
