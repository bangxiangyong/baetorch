# Provides 4 options to train the BAE
# 1. MCMC
# 2. VI
# 3. Dropout
# 4. Ensembling
# 5. VAE (special case of VI?)

# Layer types
# 1. Dense
# 2. Conv2D
# 3. Conv2DTranspose

# Activation layers
# Sigmoid, relu , etc
# Option to configure Last layer

# Parameters of specifying model
# Encoder
# Latent
# Decoder-MU
# Decoder-SIG
# Cluster (TBA)

# Specifying model flow
# 1. specify architecture for Conv2D (encoder)
# 2. specify architecture for Dense (encoder) #optional
# 3. specify architecture for Dense (latent)
# 4. specify architecture for Dense (decoder) #optional
# 5. specify architecture for Conv2D (decoder)
# since decoder and encoder are symmetrical, end-user probably just need to specify encoder architecture
import copy

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn import Parameter, GaussianNLLLoss

# from .sparse_ae import SparseAutoencoderModule
from ..evaluation import calc_auroc
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
from piqa import SSIM, MS_SSIM, HaarPSI, MDSI, PSNR
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
from ..util.seed import bae_set_seed
from ..util.sghmc import SGHMC
from ..util.truncated_gaussian import TruncatedNormal


class AutoencoderModule(torch.nn.Module):
    def __init__(
        self,
        chain_params=[],
        last_activation="sigmoid",
        last_norm=True,
        twin_output=False,
        twin_params={"activation": "none", "norm": False},
        use_cuda=False,
        skip=False,
        homoscedestic_mode="none",
    ):
        super(AutoencoderModule, self).__init__()

        # save parameters
        self.last_activation = last_activation
        self.last_norm = last_norm
        self.twin_output = twin_output
        self.skip = skip
        self.homoscedestic_mode = homoscedestic_mode

        # define method to create type of chain
        self.init_create_chain_func()

        # specify encoder params
        self.enc_params = copy.deepcopy(chain_params)
        self.enc_params[-1].update({"last_norm": self.enc_params[-1]["norm"]})

        # determine whether conv-linear architecture is used
        if (
            len(self.enc_params) == 2
            and (
                self.enc_params[0]["base"] == "conv2d"
                or self.enc_params[0]["base"] == "conv1d"
            )
            and self.enc_params[1]["base"] == "linear"
        ):
            self.conv_linear_type = True
        else:
            self.conv_linear_type = False

        # specify decoder params
        # decoder params are reverse of enc params, with transpose enforced to be True
        # and include params related to the last layer such as last_activation, last_norm, twin_output
        self.dec_params = copy.deepcopy(chain_params)
        self.dec_params.reverse()
        for param in self.dec_params:
            param.update({"transpose": True})
        self.dec_params[-1].update(
            {
                "last_activation": last_activation,
                "last_norm": last_norm,
                "twin_output": twin_output,
                "twin_params": twin_params,
            }
        )
        self.dec_params[-1].update({"last_dropout": 0})  # disable last dropout
        self.instantiate_encoder(chain_params=self.enc_params)
        self.instantiate_decoder(chain_params=self.dec_params)

        # save number of blocks present in both and encoder/decoder
        self.num_enc_blocks = len(self.encoder)
        self.num_dec_blocks = len(self.decoder)

        # set homoscedestic mode
        self.set_log_noise(homoscedestic_mode=self.homoscedestic_mode)

        # set cuda
        self.set_cuda(use_cuda)

    def init_create_chain_func(self):
        """
        Override this to provide custom linear or conv chain functions.
        """
        # define method to create type of chain
        self.create_chain_func = {
            "linear": create_linear_chain,
            "conv2d": create_conv_chain,
            "conv1d": create_conv_chain,
        }

    def instantiate_encoder(self, chain_params=[{"base": "linear"}]):
        # instantiate the first chain
        encoder = self.create_chain_func[chain_params[0]["base"]](**chain_params[0])

        # handle conv-linear type
        # by adding Flatten() layer
        # and making sure the linear input matches the flattened conv layer
        if self.conv_linear_type:
            conv_params = chain_params[0].copy()
            linear_params = chain_params[1].copy()

            inp_dim = (
                [conv_params["input_dim"]]
                if isinstance(conv_params["input_dim"], int)
                else conv_params["input_dim"]
            )

            # get the flattened latent shapes
            (
                self.conv_latent_shapes,
                self.flatten_latent_shapes,
            ) = get_conv_latent_shapes(encoder, *inp_dim)

            linear_params.update(
                {
                    "architecture": [self.flatten_latent_shapes[-1]]
                    + linear_params["architecture"]
                }
            )
            self.dec_params[0].update(linear_params)  # update decoder params

            # append flatten layer
            encoder.append(Flatten())

            # append linear chain
            encoder = encoder + self.create_chain_func[linear_params["base"]](
                **linear_params
            )

        self.encoder = torch.nn.Sequential(*encoder)

    def instantiate_decoder(self, chain_params=[{"base": "linear"}]):
        # instantiate the first chain
        decoder = self.create_chain_func[chain_params[0]["base"]](**chain_params[0])

        # handle conv-linear type
        # by adding Reshape() layer
        if self.conv_linear_type:
            decoder.append(Reshape(self.conv_latent_shapes[-1]))
            decoder = decoder + self.create_chain_func[chain_params[1]["base"]](
                **chain_params[1]
            )

        self.decoder = torch.nn.Sequential(*decoder)

    def forward(self, x):
        if self.skip:
            return self.forward_skip(x)
        else:
            x = self.encoder(x)
            return self.decoder(x)

    def forward_skip(self, x):
        # implement skip connections from encoder to decoder
        enc_outs = []

        # collect encoder outputs
        for enc_i, block in enumerate(self.encoder):
            x = block(x)

            # collect output of encoder-blocks if it is not the last, and also
            # a valid Sequential block (unlike flatten/reshape)
            if enc_i != self.num_enc_blocks - 1 and isinstance(
                block, torch.nn.Sequential
            ):
                enc_outs.append(x)

        # reverse the order to add conveniently to the decoder-blocks outputs
        enc_outs.reverse()

        # now run through decoder-blocks
        # we apply the encoder-blocks output to the decoder blocks' inputs.
        # while ignoring the first decoder block
        skip_i = 0
        for dec_i, block in enumerate(self.decoder):
            if (
                dec_i != 0
                and isinstance(block, torch.nn.Sequential)
                or isinstance(block, TwinOutputModule)
            ):
                x += enc_outs[skip_i]
                skip_i += 1
            x = block(x)

        return x

    def reset_parameters(self):
        self._reset_nested_parameters(self.encoder)
        self._reset_nested_parameters(self.decoder)

        return self

    def _reset_parameters(self, child_layer):
        if hasattr(child_layer, "reset_parameters"):
            child_layer.reset_parameters()

    def _reset_nested_parameters(self, network):
        if hasattr(network, "children"):
            for child_1 in network.children():
                for child_2 in child_1.children():
                    self._reset_parameters(child_2)
                    for child_3 in child_2.children():
                        self._reset_parameters(child_3)
                        for child_4 in child_3.children():
                            self._reset_parameters(child_4)
        return network

    def set_child_cuda(self, child, use_cuda=False):
        if isinstance(child, torch.nn.Sequential) or isinstance(
            child, TwinOutputModule
        ):
            for child in child.children():
                child.use_cuda = use_cuda
        else:
            child.use_cuda = use_cuda

    def set_cuda(self, use_cuda=False):
        self.set_child_cuda(self.encoder, use_cuda)
        self.set_child_cuda(self.decoder, use_cuda)

        self.use_cuda = use_cuda

        if use_cuda:
            self.cuda()
        else:
            self.cpu()

    def get_input_dimensions(self):
        """
        Returns total flattened input dimension
        """
        first_layer = self.enc_params[0]
        if first_layer["base"] == "conv2d" or first_layer["base"] == "conv1d":
            total_input_dim = first_layer["conv_channels"][0] * np.product(
                first_layer["input_dim"]
            )
        elif first_layer["base"] == "linear":
            total_input_dim = np.product(first_layer["architecture"][0])
        return total_input_dim

    def set_log_noise(self, homoscedestic_mode=None, init_log_noise=1e-3):
        """
        For homoscedestic regression, sets the dimensions of free parameter `log_noise`.
        We assume three possible cases :
        - "single" : size of 1
        - "every" : size equals to output dimensions
        - "none" : not inferred, and its exact value is set to 1. This is causes the neg log likelihood loss to be equal to MSE.
        """

        if homoscedestic_mode is None:
            homoscedestic_mode = self.homoscedestic_mode

        if homoscedestic_mode == "single":
            self.log_noise = Parameter(torch.FloatTensor([np.log(init_log_noise)] * 1))

        elif homoscedestic_mode == "every":
            log_noise_size = self.get_input_dimensions()
            self.log_noise = Parameter(
                torch.FloatTensor([np.log(init_log_noise)] * log_noise_size)
            )
        else:
            self.log_noise = Parameter(torch.FloatTensor([[0.0]]), requires_grad=False)


class BAE_BaseClass:
    def __init__(
        self,
        chain_params=[],
        last_activation="sigmoid",
        last_norm=None,
        twin_output=False,
        twin_params={"activation": "none", "norm": "none"},
        use_cuda=False,
        skip=False,
        homoscedestic_mode="none",
        num_samples=1,
        anchored=False,
        weight_decay=1e-10,
        num_epochs=10,
        verbose=True,
        learning_rate=0.01,
        model_type="deterministic",
        model_name="BAE",
        scheduler_enabled=False,
        likelihood="gaussian",
        AE_Module=AutoencoderModule,
        scaler=TorchMinMaxScaler,
        scaler_enabled=False,
        sparse_scale=0,
        l1_prior=False,
        stochastic_seed=-1,
        mean_prior_loss=False,
        collect_grads=False,
        collect_losses=True,
        **ae_params,
    ):
        # save kwargs
        self.num_samples = num_samples
        self.anchored = anchored
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.use_cuda = use_cuda
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.model_name = model_name
        self.losses = []
        self.scheduler_enabled = scheduler_enabled
        self.likelihood = likelihood
        self.num_iterations = 1
        self.homoscedestic_mode = homoscedestic_mode
        self.scaler_enabled = scaler_enabled
        self.skip = skip
        self.twin_params = twin_params
        self.twin_output = twin_output
        self.last_norm = last_norm
        self.last_activation = last_activation
        self.chain_params = chain_params
        self.sparse_scale = sparse_scale
        self.activ_loss = False
        self.stochastic_seed = stochastic_seed
        self.mean_prior_loss = mean_prior_loss
        self.collect_grads = collect_grads
        self.collect_losses = collect_losses

        # check if forwarding model includes activation loss
        if self.sparse_scale > 0:
            self.activ_loss = True

        # if sparse autoencoder is preferred
        if self.sparse_scale > 0:
            AE_Module = SparseAutoencoderModule

        # if l1_prior is used instead of l2
        self.l1_prior = l1_prior

        # init AE module
        self.autoencoder = self.init_autoencoder_module(
            AE_Module=AE_Module,
            homoscedestic_mode=homoscedestic_mode,
            chain_params=chain_params,
            last_activation=last_activation,
            last_norm=last_norm,
            twin_output=twin_output,
            twin_params=twin_params,
            use_cuda=use_cuda,
            skip=skip,
            **ae_params,
        )

        # set optimisers
        self.optimisers = []
        self.losses = []
        self.grads = []

        # init scaler
        if self.scaler_enabled:
            self.scaler = scaler()

        # init anchored weights prior
        self.init_anchored_weight()

        # init ssim
        if self.likelihood == "ssim":
            self.ssim = (
                torch.nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
                if use_cuda
                else torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            )

        elif self.likelihood == "gaussian_v2":
            self.gauss_loss = (
                torch.nn.GaussianNLLLoss(reduction="none").cuda()
                if use_cuda
                else torch.nn.GaussianNLLLoss(reduction="none")
            )

        # collect name
        self.layer_names = []
        if type(self.autoencoder) == list:
            temp_ac = self.autoencoder[0]
        else:
            temp_ac = self.autoencoder
        for n, p in temp_ac.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                self.layer_names.append(n)

    def init_autoencoder_module(self, AE_Module, *args, **params):
        """
        Override this if required to init different AE Module.
        """
        return AE_Module(*args, **params)

    def init_fit(self):
        # init optimisers and scheduler for fitting model
        if len(self.optimisers) == 0:
            self.set_optimisers()
        else:
            self.load_optimisers_state()
        if self.scheduler_enabled:
            self.init_scheduler()

    def fit(self, x, y=None, num_epochs=5, init_fit=True, **fit_kwargs):
        # initialise optimisers and scheduler (if applicable)
        if init_fit:
            self.init_fit()

        # handle train loader
        if isinstance(x, torch.utils.data.dataloader.DataLoader):

            # if scaler is enabled
            # fit transform on it first
            if self.scaler_enabled:
                self.scaler.fit(x.dataset.x)

            for epoch in tqdm(range(num_epochs)):
                temp_loss = []

                # keep track of current number of gradients and average them
                # if self.collect_grads:
                #     len_grad = len(self.grads)
                for batch_idx, (data, target) in enumerate(x):
                    if len(data) <= 2:
                        continue
                    else:
                        # handle scaler
                        if self.scaler_enabled:
                            loss = self.fit_one(
                                x=self.scaler.transform(data), y=y, **fit_kwargs
                            )
                        else:
                            loss = self.fit_one(x=data, y=y, **fit_kwargs)

                    if self.collect_losses:
                        temp_loss.append(loss)
                if self.collect_losses:
                    self.losses.append(np.mean(temp_loss))  # indent?
                    self.print_loss(epoch, self.losses[-1])
                else:
                    self.print_loss(epoch, loss)

                # EXPERIMENTAL: collect gradients if needed
                if self.collect_grads:
                    temp_grads = []
                    for n, p in self.autoencoder[0].named_parameters():
                        if (p.requires_grad) and ("bias" not in n):
                            temp_grads.append(
                                p.grad.abs().mean().detach().cpu().numpy()
                            )
                    self.grads.append(temp_grads)

        # handle np.ndarray or tensor
        else:
            # fit scaler
            if self.scaler_enabled:
                x = self.scaler.fit_transform(x)

            # convert to tensor
            x, y = self.convert_tensor(x=x, y=y)

            # start running through loop
            for epoch in tqdm(range(num_epochs)):
                loss = self.fit_one(x=x, y=y, **fit_kwargs)

                if self.collect_losses:
                    self.losses.append(loss)
                    self.print_loss(epoch, self.losses[-1])

                # EXPERIMENTAL: collect gradients if needed
                if self.collect_grads:
                    temp_grads = []
                    for n, p in self.autoencoder[0].named_parameters():
                        if (p.requires_grad) and ("bias" not in n):
                            temp_grads.append(
                                p.grad.abs().mean().detach().cpu().numpy()
                            )
                    self.grads.append(temp_grads)

        # update scaler
        if self.scaler_enabled:
            self.scaler.fitted = True

        return self

    def fit_one(self, x, y=None, y_scaler=0.01):
        """
        Template for vanilla fitting, developers are very likely to override this to provide custom fitting functions.
        """
        # extract input and output size from data
        # and convert into tensor, if not already
        x, y = self.convert_tensor(x, y)

        # train for n epochs
        # loss = self.criterion(autoencoder=self.autoencoder, x=x, y=y)

        # EXPERIMENTAL
        if y is None:
            loss = self.criterion(autoencoder=self.autoencoder, x=x, y=y)
        else:  # semi-supervised learning
            # loss = (
            #     self.criterion(autoencoder=self.autoencoder, x=x, y=None)
            #     - self.criterion(autoencoder=self.autoencoder, x=y, y=None) * y_scaler
            # )
            loss = self.criterion(autoencoder=self.autoencoder, x=x, y=y)

        # backpropagate
        self.zero_optimisers()
        loss.backward()
        self.step_optimisers()

        # if scheduler is enabled, update it
        if self.scheduler_enabled:
            self.step_scheduler()

        if self.use_cuda:
            x.cpu()
            if y is not None:
                y.cpu()
        return loss.item()

    def fit_test(
        self, x, x_id_test, x_ood_test, y=None, num_epochs=5, every=1, **fit_kwargs
    ):
        # initialise optimisers and scheduler (if applicable)
        self.init_fit()
        self.aurocs = []

        # handle train loader
        if isinstance(x, torch.utils.data.dataloader.DataLoader):

            # if scaler is enabled
            # fit transform on it first
            if self.scaler_enabled:
                self.scaler.fit(x.dataset.x)

            for epoch in tqdm(range(num_epochs)):
                temp_loss = []
                for batch_idx, (data, target) in enumerate(x):
                    if len(data) <= 2:
                        continue
                    else:
                        # handle scaler
                        if self.scaler_enabled:
                            loss = self.fit_one(
                                x=self.scaler.transform(data), y=y, **fit_kwargs
                            )
                        else:
                            loss = self.fit_one(x=data, y=y, **fit_kwargs)
                    temp_loss.append(loss)
                self.losses.append(np.mean(temp_loss))  # indent?

                if epoch % every == 0:
                    # predict
                    nll_id = (
                        self.predict(x_id_test, select_keys=["nll"])["nll"]
                        .mean(0)
                        .mean(-1)
                    )
                    nll_ood = (
                        self.predict(x_ood_test, select_keys=["nll"])["nll"]
                        .mean(0)
                        .mean(-1)
                    )
                    self.aurocs.append(calc_auroc(nll_id.mean(-1), nll_ood.mean(-1)))

                self.print_loss(epoch, self.losses[-1])

        # handle np.ndarray or tensor
        else:
            # fit scaler
            if self.scaler_enabled:
                x = self.scaler.fit_transform(x)

            # convert to tensor
            x, y = self.convert_tensor(x=x, y=y)

            # start running through loop
            for epoch in tqdm(range(num_epochs)):
                loss = self.fit_one(x=x, y=y, **fit_kwargs)
                self.losses.append(loss)
                self.print_loss(epoch, self.losses[-1])

                if epoch % every == 0:
                    # predict
                    nll_id = (
                        self.predict(x_id_test, select_keys=["nll"])["nll"]
                        .mean(0)
                        .mean(-1)
                    )
                    nll_ood = (
                        self.predict(x_ood_test, select_keys=["nll"])["nll"]
                        .mean(0)
                        .mean(-1)
                    )
                    self.aurocs.append(calc_auroc(nll_id.mean(-1), nll_ood.mean(-1)))

        # update scaler
        if self.scaler_enabled:
            self.scaler.fitted = True

        return self

    def criterion(self, autoencoder, x, y=None):
        """
        Override this if necessary.
        This computes the combined loss to be optimised.
        Prior and Likelihood losses are to be computed here.
        """
        # EXPERIMENTAL : SEMI SUPERVISED
        if y is not None:
            # working code::
            ae_outputs = autoencoder(x)
            y_pred_mu, y_pred_sig, activ_loss_ = self.unpack_ae_outputs(
                ae_outputs, autoencoder.log_noise
            )
            id_nll = self.log_likelihood_loss(
                y_pred_mu, x, y_pred_sig, return_mean=False
            ).mean(-1)

            # # get second prediction
            ae_outputs2 = autoencoder(y)

            # unpack AE outputs due to AE's nature of multiple outputs
            y_pred_mu_2, y_pred_sig_2, activ_loss_2 = self.unpack_ae_outputs(
                ae_outputs2, autoencoder.log_noise
            )

            ood_nll = self.log_likelihood_loss(
                y_pred_mu_2, y, y_pred_sig_2, return_mean=False
            ).mean(-1)

            # classification
            class_loss = F.binary_cross_entropy_with_logits(
                id_nll, torch.zeros_like(id_nll), reduction="mean"
            ) + F.binary_cross_entropy_with_logits(
                ood_nll, torch.ones_like(ood_nll), reduction="mean"
            )

            nll = id_nll.mean() + class_loss

        # UNSUPERVISED
        else:
            # get AE forwards
            ae_outputs = autoencoder(x)

            # unpack AE outputs due to AE's nature of multiple outputs
            y_pred_mu, y_pred_sig, activ_loss_ = self.unpack_ae_outputs(
                ae_outputs, autoencoder.log_noise
            )
            nll = self.log_likelihood_loss(y_pred_mu, x, y_pred_sig, return_mean=True)

        # check if activation loss is needed
        if self.activ_loss and self.sparse_scale > 0:
            nll = nll + activ_loss_ * self.sparse_scale

        # prior loss
        if self.weight_decay > 0:
            prior_loss = self.log_prior_loss(model=autoencoder)
            return nll + prior_loss
        else:
            return nll

    def predict_one(
        self,
        x,
        y=None,
        select_keys=["y_mu", "y_sigma", "se", "bce", "nll"],
        autoencoder_=None,
    ):
        """
        Predict once on a batch of data
        """
        # transform using scaler
        if self.scaler_enabled:
            x = self.scaler.transform(x)

        # convert to tensor
        x, y = self.convert_tensor(x=x, y=y)

        # forward ae
        if autoencoder_ is None:
            autoencoder_ = self.autoencoder

        ae_outputs = autoencoder_(x)

        # unpack ae outputs
        # note : activ loss is not required in prediction
        y_pred_mu, y_pred_sig, _ = self.unpack_ae_outputs(
            ae_outputs, autoencoder_.log_noise
        )

        # check if reshape is needed
        if len(x.shape) > 2 and self.likelihood != "ssim":
            # save original shape
            reshape = list(x.shape)

            # flatten inputs
            y_pred_mu = flatten_torch(y_pred_mu)
            x = flatten_torch(x)
            y_pred_sig = (
                flatten_torch(y_pred_sig) if len(y_pred_sig.shape) > 2 else y_pred_sig
            )
        else:
            reshape = None

        # start collecting predictions
        res = {}
        for key in select_keys:
            if key == "y_mu":
                res.update({key: y_pred_mu})
            elif key == "y_sigma":
                res.update({key: torch.exp(y_pred_sig)})
            elif key == "se":
                res.update({key: torch.pow(y_pred_mu - x, 2)})
            elif key == "bce":
                res.update(
                    {key: F.binary_cross_entropy(y_pred_mu, x, reduction="none")}
                )
            elif key == "nll":
                res.update({key: self._nll(y_pred_mu, x, y_pred_sig)})

        # convert back to numpy
        # check if reshape is needed
        if reshape is not None:
            # recover original shape
            res = {
                key: val.view(*reshape).detach().cpu().numpy()
                if key != "y_sigma"
                else val.detach().cpu().numpy()
                for key, val in res.items()
            }
        else:
            res = {key: val.detach().cpu().numpy() for key, val in res.items()}

        return res

    def predict_dataloader(
        self, dataloader, select_keys, autoencoder_, aggregate=False, y=None
    ):
        """
        Accumulate results from each test batch, instead of calculating all at one go.

        If aggregate, will summarise samples into mean and var, and will not return raw BAE samples.
        """

        final_results = []  # return list of prediction dicts
        for batch_idx, (data, target) in tqdm(enumerate(dataloader)):
            # predict new batch of results
            # handle type of AE model
            if self.model_type == "deterministic":
                next_batch_result_samples = [
                    self.predict_one(
                        x=data, select_keys=select_keys, autoencoder_=autoencoder_
                    )
                ]
            elif self.model_type == "stochastic":
                next_batch_result_samples = [
                    self.predict_one(
                        x=data, select_keys=select_keys, autoencoder_=autoencoder_
                    )
                    for i in range(self.num_samples)
                ]
            elif self.model_type == "list":
                next_batch_result_samples = [
                    self.predict_one(x=data, select_keys=select_keys, autoencoder_=ae_i)
                    for ae_i in autoencoder_
                ]

            # summarise samples if needed
            # to reduce total memory usage
            if aggregate:
                agg_res = self.aggregate_samples(
                    self.concat_predictions(next_batch_result_samples),
                    select_keys=select_keys,
                )
                next_batch_result_samples = agg_res

            # populate for first time
            if batch_idx == 0:
                final_results = copy.deepcopy(next_batch_result_samples)

            # append for subsequent batches
            else:
                # not-aggregated i.e. list of dict of BAE samples
                if not aggregate:
                    for i in range(self.num_samples):
                        for key in final_results[i].keys():
                            final_results[i][key] = np.concatenate(
                                (
                                    final_results[i][key],
                                    next_batch_result_samples[i][key],
                                ),
                                axis=0,
                            )
                # handle aggregated
                else:
                    for key in final_results.keys():
                        final_results[key] = np.concatenate(
                            (final_results[key], next_batch_result_samples[key]),
                            axis=0,
                        )

        return final_results

    def aggregate_samples(
        self, bae_pred, select_keys=["y_mu", "y_sigma", "se", "bce", "nll"]
    ):
        # computes mean/var of BAE sampled predictions
        # given keys e.g. "nll", returns a dict with "nll_mean", "nll_var"
        # "waic" is also calculated automatically, if key is "nll"
        # for y_mu_mean, y_mu_var, nll_mean, nll_var, waic

        final_res = {}
        for key in select_keys:
            key_mean = flatten_np(np.mean(bae_pred[key], axis=0))
            key_var = flatten_np(np.var(bae_pred[key], axis=0))

            final_res.update({key + "_mean": key_mean, key + "_var": key_var})

            if key == "nll":
                waic = key_mean + key_var
                final_res.update({"waic": waic})
        return final_res

    def predict_(self, x, *args, **kwargs):
        """
        Based on data type of x (array or dataloader) , choose to predict using the right function.
        """
        if isinstance(x, torch.utils.data.dataloader.DataLoader):
            return self.predict_dataloader(x, *args, **kwargs)
        else:
            return self.predict_one(x, *args, **kwargs)

    def predict(
        self,
        x,
        y=None,
        select_keys=["y_mu", "y_sigma", "se", "bce", "nll"],
        autoencoder_=None,
        aggregate=False,
    ):
        """
        Actual predict function exposed to user. User should be exposed to use only this predict function.
        """
        with torch.no_grad():
            # predict based on model types
            if self.model_type == "deterministic":
                return self.predict_(
                    x=x, y=y, select_keys=select_keys, autoencoder_=autoencoder_
                )

            elif (self.model_type == "stochastic") or (self.model_type == "list"):
                if self.model_type == "stochastic":
                    if self.stochastic_seed != -1:
                        bae_set_seed(self.stochastic_seed)

                # handle data loader / numpy inputs
                # type 1: data loader
                if isinstance(x, torch.utils.data.dataloader.DataLoader):
                    predictions = self.predict_dataloader(
                        x,
                        y=y,
                        select_keys=select_keys,
                        autoencoder_=self.autoencoder,
                        aggregate=aggregate,
                    )
                # type2 : numpy inputs
                else:
                    # check model type
                    # stochastic: e.g. VAE,VI,MCD
                    # list: ensemble, sghmc
                    if self.model_type == "stochastic":
                        predictions = [
                            self.predict_one(
                                x=x,
                                y=y,
                                select_keys=select_keys,
                                autoencoder_=self.autoencoder,
                            )
                            for i in range(self.num_samples)
                        ]
                    elif self.model_type == "list":
                        predictions = [
                            self.predict_one(
                                x=x,
                                y=y,
                                select_keys=select_keys,
                                autoencoder_=ae_i,
                            )
                            for ae_i in self.autoencoder
                        ]
                    # aggregate results?
                    if aggregate:
                        predictions = self.aggregate_samples(
                            bae_pred=self.concat_predictions(predictions),
                            select_keys=select_keys,
                        )
                # return results
                # return raw BAE samples of results or aggregated ones
                if not aggregate:
                    return self.concat_predictions(predictions)
                else:
                    return predictions

    def log_prior_loss(self, model):
        # check if anchored prior is used
        if self.anchored:
            mu = model.anchored_prior

            if self.use_cuda:
                mu = mu.cuda()

        weights = torch.cat([parameter.flatten() for parameter in model.parameters()])

        # OPTION 1: Use anchored loss if necessary
        # OPTION 2: Scale prior by number of parameters with mean_prior_loss
        # OPTION 3: Use L1 regularisation instead of L2 (default)
        if self.anchored:
            if self.mean_prior_loss:
                prior_loss = torch.pow((weights - mu), 2).mean() * self.weight_decay
            else:
                prior_loss = torch.pow((weights - mu), 2).sum() * self.weight_decay
        else:
            if self.mean_prior_loss:
                if self.l1_prior:
                    prior_loss = torch.abs(weights).mean() * self.weight_decay
                else:
                    prior_loss = torch.pow(weights, 2).mean() * self.weight_decay
            else:
                if self.l1_prior:
                    prior_loss = torch.abs(weights).sum() * self.weight_decay
                else:
                    prior_loss = torch.pow(weights, 2).sum() * self.weight_decay

        return prior_loss

    def init_anchored_weight(self):
        """
        Wrapper to initialise anchored weights for Bayesian ensembling.
        """
        # init anchored weights prior
        if self.anchored and self.weight_decay > 0:
            # handle ensemble type of model
            if self.model_type == "list":
                for autoencoder in self.autoencoder:
                    self.init_anchored_weight_(autoencoder)
            else:
                self.init_anchored_weight_(self.autoencoder)

    def init_anchored_weight_(self, model):
        """
        Internal method to actually init the anchored weights.
        """
        model_weights = torch.cat(
            [parameter.flatten() for parameter in model.parameters()]
        )
        anchored_prior = torch.ones_like(model_weights) * model_weights.detach()
        model.anchored_prior = anchored_prior

    def unpack_ae_outputs(self, ae_outputs, ae_log_noise):
        # first unpack activation loss
        if self.activ_loss:
            ae_outputs_, activ_loss = ae_outputs
        else:
            ae_outputs_ = ae_outputs
            activ_loss = None

        # next, unpack twin outputs if available
        if self.twin_output:
            y_pred_mu, y_pred_sig = ae_outputs_
        else:
            y_pred_sig = ae_log_noise
            y_pred_mu = ae_outputs_
        return y_pred_mu, y_pred_sig, activ_loss

    def log_likelihood_loss(self, y_pred_mu, x, y_pred_sig, return_mean=True):
        """
        Compute the log likelihood loss i.e NLL depending on the chosen likelihood distribution.
        y_pred_sig is used to parameterise distributions with two parameters e.g Gaussian and will be
        effectively ignored for single parameter distribution which doesn't require it.
        """

        # compute actual loss wrt. input data
        if len(x.shape) > 2 and self.likelihood != "ssim":
            # note : flatten y_pred_sig only when necessary
            nll_loss = self._nll(
                y_pred_mu=flatten_torch(y_pred_mu),
                y_true=flatten_torch(x),
                y_pred_sig=flatten_torch(y_pred_sig)
                if len(y_pred_sig.shape) > 2
                else y_pred_sig,
            )
        else:
            nll_loss = self._nll(
                y_pred_mu=y_pred_mu,
                y_true=x,
                y_pred_sig=y_pred_sig,
            )

        if return_mean:
            nll_loss = nll_loss.mean()
        return nll_loss

    def _nll(self, y_pred_mu, y_true, y_pred_sig=None):
        """
        Given y_pred_mu and y_pred_sig and input data (y_true) , compute the chosen likelihood.
        """
        if self.likelihood == "gaussian":
            if self.homoscedestic_mode == "none" and not self.twin_output:
                nll = (y_true - y_pred_mu) ** 2  # mse
            else:
                nll = self.log_gaussian_loss_logsigma_torch(
                    y_pred_mu, y_true, y_pred_sig
                )

        if self.likelihood == "gaussian_v2":
            if self.homoscedestic_mode == "none" and not self.twin_output:
                nll = (y_true - y_pred_mu) ** 2
            else:

                # HOMO GAUSS
                if self.use_cuda:
                    var = (
                        torch.ones(*y_true.shape, requires_grad=True).cuda()
                        * y_pred_sig
                    )
                else:
                    var = torch.ones(*y_true.shape, requires_grad=True) * y_pred_sig

                nll = self.gauss_loss(
                    y_pred_mu, y_true, torch.nn.functional.elu(var) + 1
                )
                # nll = self.gauss_loss(
                #     y_pred_mu, y_true, torch.nn.functional.softplus(var)
                # )

        elif self.likelihood == "laplace":
            if self.homoscedestic_mode == "none" and not self.twin_output:
                nll = torch.abs(y_pred_mu - y_true)
            else:
                nll = self.log_laplace_loss_torch(y_pred_mu, y_true, y_pred_sig)

        elif self.likelihood == "bernoulli":
            nll = F.binary_cross_entropy(y_pred_mu, y_true, reduction="none")
        elif self.likelihood == "cbernoulli":
            nll = self.log_cbernoulli_loss_torch(y_pred_mu, y_true)
        elif self.likelihood == "truncated_gaussian":
            if self.homoscedestic_mode == "none" and not self.twin_output:
                nll = self.log_truncated_loss_torch(
                    y_pred_mu, y_true, torch.ones_like(y_pred_mu)
                )
            else:
                nll = self.log_truncated_loss_torch(
                    y_pred_mu, y_true, torch.nn.functional.elu(y_pred_sig) + 1
                )
                # nll = self.log_truncated_loss_torch(
                #     y_pred_mu, y_true, torch.nn.functional.softplus(y_pred_sig)
                # )
        elif self.likelihood == "ssim":
            nll = 1 - self.ssim(y_pred_mu, y_true)
        elif self.likelihood == "beta":
            nll = self.log_beta_loss_torch(y_pred_mu, y_true, y_pred_sig)
        return nll

    def log_gaussian_loss_logsigma_torch(self, y_pred, y_true, log_sigma):
        neg_log_likelihood = (
            ((y_true - y_pred) ** 2) * torch.exp(-log_sigma) * 0.5
        ) + (0.5 * log_sigma)

        return neg_log_likelihood

    def log_cbernoulli_loss_torch(self, y_pred_mu, y_true):
        if hasattr(self, "cb") == False:
            self.cb = CB_Distribution()
        nll_cb = self.cb.log_cbernoulli_loss_torch(y_pred_mu, y_true, mode="non-robert")
        return nll_cb

    def log_laplace_loss_torch(self, y_pred_mu, y_true, y_pred_sig):
        nll = torch.abs(y_pred_mu - y_true) * torch.exp(-y_pred_sig) + y_pred_sig
        return nll

    def log_truncated_loss_torch(self, y_pred_mu, y_true, y_pred_sig):
        trunc_g = TruncatedNormal(loc=y_pred_mu, scale=y_pred_sig, a=0.0, b=1.0)
        nll_trunc_g = -trunc_g.log_prob(y_true)
        return nll_trunc_g

    def log_beta_loss_torch(self, y_pred_mu, y_true, y_pred_sig):
        beta_dist = torch.distributions.beta.Beta(
            F.softplus(y_pred_mu) + 1e-11, torch.ones_like(y_pred_sig) + 1e-11
        )
        nll_beta = -beta_dist.log_prob(y_true)
        return nll_beta

    def print_loss(self, epoch, loss):
        if self.verbose:
            print("LOSS #{}:{}".format(epoch, loss))

    def convert_tensor(self, x, y=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if y is not None:
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).float()

        # handle use cuda
        if self.use_cuda and not x.is_cuda:
            if y is None:
                return x.cuda(), None
            else:
                return x.cuda(), y.cuda()
        else:
            if y is None:
                return x, None
            else:
                return x, y

    def get_optimisers_list(self, autoencoder):
        optimiser_list = []
        optimiser_list += [{"params": autoencoder.decoder.parameters()}]
        optimiser_list += [{"params": autoencoder.encoder.parameters()}]

        if not self.twin_output:
            optimiser_list += [{"params": autoencoder.log_noise}]

        return optimiser_list

    def get_optimisers(self, autoencoder):
        optimiser_list = self.get_optimisers_list(autoencoder)
        return torch.optim.Adam(optimiser_list, lr=self.learning_rate)

    def set_optimisers(self):
        # ensemble
        if self.model_type == "list":
            self.optimisers = [self.get_optimisers(model) for model in self.autoencoder]
        # stochastic
        elif self.model_type == "stochastic":
            self.optimisers = [self.get_optimisers(self.autoencoder)]
        # deterministic
        elif self.model_type == "deterministic":
            self.optimisers = [self.get_optimisers(self.autoencoder)]
        else:
            raise NotImplemented(
                "Model type invalid. Has to be list, stochastic or deterministic."
            )
        self.save_optimisers_state()

        return self.optimisers

    def save_optimisers_state(self):
        self.saved_optimisers_state = [
            optimiser.state_dict() for optimiser in self.optimisers
        ]

    def load_optimisers_state(self):
        for optimiser, state in zip(self.optimisers, self.saved_optimisers_state):
            optimiser.load_state_dict(state)

    def init_scheduler(self, half_iterations=None, min_lr=None, max_lr=None):
        # resort to internal stored values for scheduler
        if half_iterations is None or min_lr is None or max_lr is None:
            half_iterations = self.half_iterations
            min_lr = self.min_lr
            max_lr = self.max_lr
        else:
            self.half_iterations = half_iterations
            self.min_lr = min_lr
            self.max_lr = max_lr

        # handle model type to access optimiser hierarchy
        self.scheduler = [
            torch.optim.lr_scheduler.CyclicLR(
                optimiser,
                step_size_up=half_iterations,
                base_lr=min_lr,
                max_lr=max_lr,
                cycle_momentum=False,
            )
            for optimiser in self.optimisers
        ]

        self.scheduler_enabled = True

        # set init learning rate
        self.learning_rate = self.min_lr
        self.learning_rate_sig = self.min_lr

        return self.scheduler

    def set_learning_rate(self, learning_rate=None):
        # update internal learning rate
        if learning_rate is None:
            learning_rate = self.learning_rate
        else:
            self.learning_rate = learning_rate

        # handle access to optimiser parameters
        if len(self.optimisers) > 0:
            for optimiser in self.optimisers:
                for group in optimiser.param_groups:
                    group["lr"] = learning_rate

    def zero_optimisers(self):
        for optimiser in self.optimisers:
            optimiser.zero_grad()

    def step_optimisers(self):
        for optimiser in self.optimisers:
            optimiser.step()

    def step_scheduler(self):
        for scheduler in self.scheduler:
            scheduler.step()

    def partial_fit_scaler(self, x):
        self.scaler = self.scaler.partial_fit(x)

    def init_scaler(self, scaler=MinMaxScaler):
        self.scaler = scaler()

    def concat_predictions(self, predictions):
        # get the keys of dict
        select_keys = list(predictions[0].keys())

        # stack them
        stacked_predictions = {
            key: np.concatenate(
                [np.expand_dims(pred_[key], 0) for pred_ in predictions]
            )
            for key in select_keys
        }
        return stacked_predictions

    def save_model_state(self, filename=None, folder_path="torch_model/"):
        create_dir(folder_path)
        if filename is None:
            temp = True
        if self.model_type == "list":
            for model_i, autoencoder in enumerate(self.autoencoder):
                if temp:
                    torch_filename = self.model_name + "_" + str(model_i) + ".pt"
                    torch_filename = "temp_" + torch_filename
                else:
                    torch_filename = temp
                torch.save(autoencoder.state_dict(), folder_path + torch_filename)

        else:  # stochastic model
            if temp:
                torch_filename = self.model_name + ".pt"
                torch_filename = "temp_" + torch_filename
            else:
                torch_filename = temp
            torch.save(self.autoencoder.state_dict(), folder_path + torch_filename)

    def load_model_state(self, filename=None, folder_path="torch_model/"):
        create_dir(folder_path)
        if filename is None:
            temp = True
        if self.model_type == "list":
            for model_i, autoencoder in enumerate(self.autoencoder):
                if temp:
                    torch_filename = self.model_name + "_" + str(model_i) + ".pt"
                    torch_filename = "temp_" + torch_filename
                else:
                    torch_filename = temp
                self.autoencoder[model_i].load_state_dict(
                    torch.load(folder_path + torch_filename)
                )
        else:  # stochastic model
            if temp:
                torch_filename = self.model_name + ".pt"
                torch_filename = "temp_" + torch_filename
            else:
                torch_filename = temp
            self.autoencoder.load_state_dict(torch.load(folder_path + torch_filename))

    def reset_parameters(self):
        if self.model_type == "list":
            for autoencoder in self.autoencoder:
                autoencoder.reset_parameters()
        else:
            self.autoencoder.reset_parameters()
        self.losses = []


class SparseAutoencoderModule(AutoencoderModule):
    def __init__(self, **params):
        super(SparseAutoencoderModule, self).__init__(**params)

    def forward(self, x):
        if self.skip:
            return self.forward_skip(x)
        else:
            for enc_i, block in enumerate(self.encoder):
                x = block(x)

                # handle sparse activation loss
                if isinstance(block, torch.nn.Sequential):
                    sparse_loss_new = torch.abs(x).mean()
                    if enc_i == 0:
                        sparse_loss = sparse_loss_new
                    else:
                        sparse_loss += sparse_loss_new

            for dec_i, block in enumerate(self.decoder):
                x = block(x)

                # handle sparse activation loss
                if isinstance(block, torch.nn.Sequential):
                    sparse_loss_new = torch.abs(x).mean()
                    sparse_loss += sparse_loss_new

            return [x, sparse_loss]

    def forward_skip(self, x):
        # implement skip connections from encoder to decoder
        enc_outs = []

        # collect encoder outputs
        for enc_i, block in enumerate(self.encoder):
            x = block(x)
            # handle sparse activation loss
            if isinstance(block, torch.nn.Sequential):
                sparse_loss_new = torch.abs(x).mean()
                # sparse_loss_new = torch.abs(x)
                if enc_i == 0:
                    sparse_loss = sparse_loss_new
                else:
                    sparse_loss = sparse_loss + sparse_loss_new

            # collect output of encoder-blocks if it is not the last, and also
            # a valid Sequential block (unlike flatten/reshape)
            if enc_i != self.num_enc_blocks - 1 and isinstance(
                block, torch.nn.Sequential
            ):
                enc_outs.append(x)

        # reverse the order to add conveniently to the decoder-blocks outputs
        enc_outs.reverse()

        # now run through decoder-blocks
        # we apply the encoder-blocks output to the decoder blocks' inputs.
        # while ignoring the first decoder block
        skip_i = 0
        for dec_i, block in enumerate(self.decoder):
            if (
                dec_i != 0
                and isinstance(block, torch.nn.Sequential)
                or isinstance(block, TwinOutputModule)
            ):
                x += enc_outs[skip_i]
                skip_i += 1

            x = block(x)

            if isinstance(block, torch.nn.Sequential) and (
                dec_i < (self.num_dec_blocks - 1)
            ):
                sparse_loss_new = torch.abs(x.clone()).mean()
                sparse_loss += sparse_loss_new

        return [x, sparse_loss]
