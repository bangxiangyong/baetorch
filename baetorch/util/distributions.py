from baetorch.util.seed import bae_set_seed
import torch
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
import torch.nn.functional as F
import numpy as np
import math
import copy
from torch.distributions.utils import probs_to_logits

class CB_Distribution():
    def __init__(self, probs=0):
        self.probs = probs
        self._lims = (0.499, 0.501)

    def log_bernoulli_loss(self, y_pred_mu, y_true):
        bce = -(y_true*torch.log(y_pred_mu) + (1-y_true)*torch.log(1-y_pred_mu))
        return bce

    def log_cbernoulli_loss_torch(self, y_pred_mu, y_true):
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
