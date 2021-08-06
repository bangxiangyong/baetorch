import torch
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
import torch.nn.functional as F
import numpy as np
import math
import copy

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
        # return np.where(np.logical_or(np.less(y_pred_mu, l_lim), np.greater(y_pred_mu, u_lim)), log_norm, taylor)
        return log_p


class TruncatedGaussian():
    def __init__(self, use_cuda=False, bounded_log_pdf=-100.):
        self.use_cuda = use_cuda

        # to prevent -inf log pdf
        self.bounded_log_pdf = bounded_log_pdf

    def normal_cdf_torch(self, x,x_mu=torch.tensor((0.)),x_std=torch.tensor((1.))):
        return 0.5 * (1 + torch.erf((x - x_mu) / (x_std * math.sqrt(2))))

    def normal_pdf_torch(self,x,x_mu=torch.tensor((0.)),x_std=torch.tensor((1.))):
        return torch.exp(-0.5*((x-x_mu)/x_std)**2)/(x_std*math.sqrt(2*math.pi))

    def normal_log_pdf_torch(self,x,x_mu=torch.tensor((0.)),x_std=torch.tensor((1.))):
        x_var = torch.pow(x_std,2)
        return -0.5*torch.pow(x-x_mu,2)/(x_var)-0.5*torch.log(2*math.pi*x_var)

    def truncated_normaliser(self,x_mu,x_std,a,b):
        return self.normal_cdf_torch(b,x_mu,x_std)-self.normal_cdf_torch(a,x_mu,x_std)

    def truncated_mean(self,x_mu,x_std,a,b):
        alpha = (a-x_mu)/x_std
        beta = (b-x_mu)/x_std
        return x_mu + x_std*(self.normal_pdf_torch(alpha)-self.normal_pdf_torch(beta))/self.truncated_normaliser(x_mu,x_std,a,b)

    def truncated_var(self,x_mu,x_std,a,b):
        alpha = (a-x_mu)/x_std
        beta = (b-x_mu)/x_std
        term_1 = (alpha*self.normal_pdf_torch(alpha) - beta*self.normal_pdf_torch(beta))/self.truncated_normaliser(x_mu,x_std,a,b)
        term_2 = torch.pow((self.normal_pdf_torch(alpha) - self.normal_pdf_torch(beta))/self.truncated_normaliser(x_mu,x_std,a,b),2)

        return torch.pow(x_std,2)*(1+term_1+term_2)

    def truncated_pdf(self,x,x_mu,x_std,a,b):
        res = self.normal_pdf_torch(x,x_mu,x_std)/(self.truncated_normaliser(x_mu,x_std,a,b))
        bounded_log_prob = torch.where((x > a) & (x < b),
                                    res,
                                    torch.Tensor([0.]))
        return bounded_log_prob

    def truncated_log_pdf(self,x,x_mu,x_std,a=torch.tensor((0.)),b=torch.tensor((1.))):
        bounded_log_pdf = torch.tensor((copy.copy(self.bounded_log_pdf)))
        if self.use_cuda:
            bounded_log_pdf = bounded_log_pdf.cuda()

        res = self.normal_log_pdf_torch(x,x_mu,x_std)-torch.log(self.truncated_normaliser(x_mu,x_std,a,b))

        bounded_log_prob = torch.where((x > a) & (x < b),
                                    res,
                                    bounded_log_pdf)
        return bounded_log_prob


