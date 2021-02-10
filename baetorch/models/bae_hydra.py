import copy
import numpy as np
import torch
from sklearn.decomposition import PCA

from baetorch.baetorch.models.cholesky_layer import CholLayer
from ..models.base_autoencoder import BAE_BaseClass, Autoencoder, \
    flatten_np, flatten_torch, infer_decoder, DenseLayers
from baetorch.baetorch.models.clustering_layer import ClusteringLayer

#NOTE: NEED TO TALLY NUM_SAMPLES IN AE TORCH AND BAE WRAPPER

class Hydra_Autoencoder(Autoencoder):
    """
    In BAE-Hydra, only the decoders are stochastic, the reason is to have the encoder encode into a fixed
    latent dimension to be interpretable.

    The forward function now accumulates the decoder's responses in the form of lists, before backpropagating
    through them.

    """
    def __init__(self, encoder: torch.nn.Sequential, decoder_mu=None,
                 decoder_sig=None, decoder_cluster=None, homoscedestic_mode="none",
                 use_cuda=False, num_samples=5, cluster_architecture=[],
                 num_clusters=3, decoder_last_activation="none", num_cluster_samples=0):

        self.num_cluster_samples = num_cluster_samples
        if self.num_cluster_samples ==0:
            self.decoder_cluster_enabled = False
            decoder_clusters = None
        else:
            self.decoder_cluster_enabled = True
            decoder_clusters = [ClusteringLayer(architecture=cluster_architecture,
                                                input_size=encoder.latent_dim,
                                                output_size=num_clusters) for i in range(num_cluster_samples)]
        super(Hydra_Autoencoder, self).__init__(encoder,
                                                decoder_mu=[infer_decoder(encoder, activation=encoder.activation, last_activation=decoder_last_activation) for i in range(num_samples)],
                                                decoder_sig=[copy.deepcopy(decoder_sig) for i in range(num_samples)] if decoder_sig is not None else None,
                                                homoscedestic_mode=homoscedestic_mode, use_cuda=use_cuda,
                                                decoder_cluster=decoder_clusters
                                                )
        #reset parameters of decoder sig
        if self.decoder_sig_enabled:
            for decoder_sig_ in self.decoder_sig:
                self._reset_nested_parameters(decoder_sig_)

            #overwrite decoder_full_cov status
            if isinstance(decoder_sig, CholLayer):
                self.decoder_full_cov_enabled = True
            else:
                self.decoder_full_cov_enabled = False

    def forward(self, x):
        encoded = self.encoder(x)
        decoded_mu = [decoder(encoded) for decoder in self.decoder_mu]
        decoded_list = [decoded_mu]

        if self.decoder_sig_enabled:
            decoded_sig = [decoder(encoded) for decoder in self.decoder_sig]
            decoded_list.append(decoded_sig)

        if self.decoder_cluster_enabled:
            decoded_cluster = [decoder_cluster(encoded) for decoder_cluster in self.decoder_cluster]
            decoded_list.append(decoded_cluster)
        return tuple(decoded_list)

    def set_cuda(self, use_cuda=False):

        self.set_child_cuda(self.encoder,use_cuda)

        #handle multiple decoders
        for decoder in self.decoder_mu:
            self.set_child_cuda(decoder,use_cuda)

            if use_cuda:
                decoder.cuda()
            else:
                decoder.cpu()

        #decoder sig
        if self.decoder_sig_enabled:
            for decoder_sig in self.decoder_sig:
                self.set_child_cuda(decoder_sig,use_cuda)
            if use_cuda:
                decoder_sig.cuda()
            else:
                decoder_sig.cpu()

        #decoder cluster
        if self.decoder_cluster_enabled:
            for decoder_cluster in self.decoder_cluster:
                self.set_child_cuda(decoder_cluster,use_cuda)
            if use_cuda:
                decoder_cluster.cuda()
            else:
                decoder_cluster.cpu()

        self.use_cuda = use_cuda

        if use_cuda:
            self.cuda()
        else:
            self.cpu()

    def reset_parameters(self):
        self._reset_nested_parameters(self.encoder)
        for decoder_mu in self.decoder_mu:
            self._reset_nested_parameters(decoder_mu)
        if self.decoder_sig_enabled:
            for decoder_sig in self.decoder_sig:
                self._reset_nested_parameters(decoder_sig)
        if self.decoder_cluster_enabled:
            for decoder_cluster in self.decoder_cluster:
                self._reset_nested_parameters(decoder_cluster)

        return self

#Probabilistic decoders only
class BAE_Hydra(BAE_BaseClass):
    def __init__(self,*args, autoencoder:Hydra_Autoencoder, model_name="BAE_Ensemble", cluster_weight=1000., **kwargs):
        super(BAE_Hydra, self).__init__(*args,autoencoder=autoencoder, model_name=model_name, model_type="stochastic", **kwargs)
        self.num_cluster_samples = autoencoder.num_cluster_samples
        self.cluster_weight = cluster_weight

    def init_autoencoder(self, autoencoder):
        """
        Initialise an ensemble of autoencoders
        """
        self.autoencoder = super(BAE_Hydra, self).init_autoencoder(autoencoder)

        return self.autoencoder

    def nll(self, autoencoder: Hydra_Autoencoder, x, y=None, mode="mu"):
        """
        Computes the NLL with the autoencoder, with the given likelihood.
        This depends on the given `mode` of whether it is hetero- or homoscedestic
        And also whether a decoder_sigma is enabled in the autoencoder.
        For now, mode="sigma" is only implemented for Gaussian likelihood and will only make sense in that setting.

        if y is provided,
        autoencoder : Autoencoder
            Autoencoder to compute the NLL with
        x : torch.Tensor
            input data
        y : torch.Tensor
            target data
        mode : str of "sigma" or "mu"
            If "sigma", the output of the autoencoder's decoder_mu and decoder_sigma will be used for the likelihood.
            If "mu", the autoencoder's decoder_mu and the homoscedestic term `log_noise` will be used for the likelihood.
        """
        # likelihood
        if y is None:
            y = x
        # get outputs of autoencoder, depending on the number of decoders
        # that is, if decoder sigma is enabled or not

        # ae_output = autoencoder(x)
        # y_pred_mus = ae_output[0]
        # if self.decoder_sigma_enabled:
        #     y_pred_sigs = ae_output[self.decoder_sig_index]

        if self.decoder_sigma_enabled and self.decoder_cluster_enabled:
            y_pred_mus, y_pred_sigs, y_pred_cluster = autoencoder(x)
        elif self.decoder_sigma_enabled:
            y_pred_mus, y_pred_sigs = autoencoder(x)
        elif self.decoder_cluster_enabled:
            y_pred_mus, y_pred_cluster = autoencoder(x)
        else:
            y_pred_mus = autoencoder(x)[0]

        # depending on the mode, we compute the nll
        if mode == "sigma":
            nll = torch.stack([self._nll(flatten_torch(y_pred_mus[i]), flatten_torch(y), y_pred_sigs[i]) for i in range(self.num_samples)])
        elif mode == "mu":
            #temporary switch for handling full gaussian mode
            if self.likelihood == "full_gaussian":
                self.likelihood = "gaussian"
                nll = torch.stack([self._nll(flatten_torch(y_pred_mus[i]), flatten_torch(y), autoencoder.log_noise) for i in range(self.num_samples)])
                self.likelihood = "full_gaussian"
            else:
                nll = torch.stack([self._nll(flatten_torch(y_pred_mus[i]), flatten_torch(y), autoencoder.log_noise) for i in range(self.num_samples)])

        # determine whether we have decoder cluster enabled
        if self.decoder_cluster_enabled:
            cluster_loss = torch.stack([self.cluster_loss(y_pred_cluster[i]) for i in range(len(y_pred_cluster))])
            return nll.mean() + (self.cluster_weight *cluster_loss).sum()
        else:
            return nll.mean()

    def criterion(self, autoencoder: Hydra_Autoencoder, x,y=None, mode="sigma"):
        #likelihood
        stacked_nll = self.nll(autoencoder,x,y,mode)
        # stacked_nll = stacked_nll.mean()
        # print(stacked_nll.shape)

        #prior loss
        prior_loss = self.log_prior_loss(model=autoencoder)
        prior_loss = prior_loss.mean()
        prior_loss *= self.weight_decay

        return stacked_nll + prior_loss

    def get_optimisers_list(self, autoencoder: Hydra_Autoencoder, mode="mu", sigma_train="separate"):
        optimiser_list = []
        if mode =="sigma":
            if autoencoder.decoder_sig_enabled:
                for decoder_sig in autoencoder.decoder_sig:
                    optimiser_list.append({'params':decoder_sig.parameters(),'lr':self.learning_rate_sig})

                if sigma_train == "joint": #for joint training
                    optimiser_list.append({'params':autoencoder.encoder.parameters()})
                    for decoder in autoencoder.decoder_mu:
                        optimiser_list.append({'params':decoder.parameters()})
                    optimiser_list.append({'params':autoencoder.log_noise})
        else:
            optimiser_list.append({'params':autoencoder.encoder.parameters()})
            #handle hydra autoencoder's multiple decoders
            #add each into optimiser list
            for decoder in autoencoder.decoder_mu:
                optimiser_list.append({'params':decoder.parameters()})
            optimiser_list.append({'params':autoencoder.log_noise})

        if self.decoder_cluster_enabled:
            for decoder_cluster in autoencoder.decoder_cluster:
                optimiser_list.append({'params':decoder_cluster.parameters()})

        return optimiser_list

    def _predict_samples(self,x, model_type=0, select_keys=["y_mu","y_sigma","se","bce","cbce","nll_homo","nll_sigma"]):
        #handle different model types
        y_preds = self.calc_batch(self.autoencoder, x, select_keys)

        return y_preds

    def calc_batch(self, hydra_autoencoder : Hydra_Autoencoder, x, select_keys):
        ae_output = hydra_autoencoder(x)
        y_mus = ae_output[0]
        if self.decoder_sigma_enabled:
            if self.decoder_full_cov_enabled:
                y_sigmas = [(y_sig[0].detach().cpu().numpy(),y_sig[1].detach().cpu().numpy()) for y_sig in ae_output[self.decoder_sig_index]]
            else:
                y_sigmas = [y_sig.detach().cpu().numpy() for y_sig in ae_output[self.decoder_sig_index]]
        else:
            y_sigmas = [torch.ones_like(y_mus[0]) for i in range(self.num_samples)]
        log_noise = hydra_autoencoder.log_noise.detach().cpu().numpy()
        x = flatten_np(x.detach().cpu().numpy())
        batch_output = []
        for i in range(self.num_samples):
            batch_output.append(self._calc_output_single(x,flatten_np(y_mus[i].detach().cpu().numpy()),
                                                               y_sigmas[i],
                                                               log_noise,select_keys))
        return batch_output

    def predict_cluster(self, x, encode=True):
        x = self.convert_tensor(x)
        if encode:
            return np.array([self.autoencoder.decoder_cluster[i](self.autoencoder.encoder(x)).detach().cpu().numpy() for i in range(self.num_cluster_samples)])
        else:
            return np.array([self.autoencoder.decoder_cluster[i](x).detach().cpu().numpy() for i in range(self.num_cluster_samples)])

    #TEMP
    def transpose_x(self, x):
        if len(x.shape) == 3:
            x = x.transpose(1, 2)
        elif len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)
        return x

    def convert_tensor(self, x, y=None):
        if y is None:
            x = super(BAE_Hydra, self).convert_tensor(x, y)
            return self.transpose_x(x)
        else:
            x, y = super(BAE_Hydra, self).convert_tensor(x, y)
            return self.transpose_x(x), y

    def predict_latent(self, x, transform_pca=True):
        """
        Since Hydra BAE has non-probabilistic encoder, we can obtain mean but not
        variance of the latent dimensions for each data
        """

        x = self.convert_tensor(x)
        latent_data = self.autoencoder.encoder(x).detach().cpu().numpy()

        if transform_pca:
            pca = PCA(n_components=2)
            latent_data = pca.fit_transform(latent_data)
        return latent_data, np.zeros_like(latent_data)




