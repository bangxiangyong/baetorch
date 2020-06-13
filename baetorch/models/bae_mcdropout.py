from ..models.base_autoencoder import ConvLayers, DenseLayers, BAE_BaseClass, Autoencoder
import torch

#Layers
class ConvDropoutLayers(ConvLayers):
    def __init__(self, dropout_p=0.1, **kwargs):
        super(ConvDropoutLayers, self).__init__(**kwargs)

        if self.use_cuda:
            self.dropout_layer = torch.nn.Dropout(p=dropout_p).cuda()
        else:
            self.dropout_layer = torch.nn.Dropout(p=dropout_p)

    def forward(self,x):
        #apply relu
        for layer_index,layer in enumerate(self.layers):
            #apply dropout to all layers except the last layer
            if layer_index == (len(self.layers)-1) and self.upsampling == True:
                x = layer(x)
            else:
                x = self.dropout_layer(layer(x))
        return x

class DenseDropoutLayers(DenseLayers):
    def __init__(self, dropout_p=0.1, **kwargs):
        super(DenseDropoutLayers, self).__init__(**kwargs)

        if self.use_cuda:
            self.dropout_layer = torch.nn.Dropout(p=dropout_p).cuda()
        else:
            self.dropout_layer = torch.nn.Dropout(p=dropout_p)

    def forward(self,x):
        #apply relu
        for layer_index,layer in enumerate(self.layers):
            if layer_index ==0:
                #first layer
                x = layer(x)
            else:
                #other than first layer
                x = layer(self.activation_layer(self.dropout_layer(x)))
        return x

#Model Manager
class BAE_MCDropout(BAE_BaseClass):
    def __init__(self,*args, model_name="BAE_MCDropout", num_train_samples=5,dropout_p=0.1, alpha=1., **kwargs):
        self.num_train_samples = num_train_samples #for training averaging
        self.dropout_p = dropout_p
        self.alpha = alpha
        super(BAE_MCDropout, self).__init__(*args, model_name=model_name, model_type="stochastic", **kwargs)

    def log_prior_loss(self, model, mu=torch.Tensor([0.]), L=2):
        prior_loss = super(BAE_MCDropout, self).log_prior_loss(model,mu,L)
        prior_loss *= (1.0 - self.dropout_p)
        return prior_loss

    def criterion(self, autoencoder, x,y=None, mode="mu"):
        """
        `autoencoder` here is a list of autoencoder
        We sum the losses and backpropagate them at one go
        """
        stacked_criterion = torch.stack([super(BAE_MCDropout, self).criterion(self.autoencoder, x,y=y, mode=mode) for i in range(self.num_train_samples)])
        return stacked_criterion.mean()

    def convert_conv_dropout(self,conv_layer):
        conv_params = {}
        for key, val in conv_layer.__dict__.items():
            exclude_params = ["activation_layer", "model_kwargs","training", "conv2d_layer_type","conv2d_trans_layer_type"]
            if key[0] != '_' and key not in exclude_params:
                conv_params.update({key:val})
        conv_dropout = ConvDropoutLayers(**conv_params, dropout_p=self.dropout_p, reverse_params=False)
        return conv_dropout

    def convert_dense_dropout(self,dense_layer):
        dense_params = {}
        for key, val in dense_layer.__dict__.items():
            exclude_params = ["activation_layer", "model_kwargs"]
            if key[0] != '_' and key not in exclude_params:
                dense_params.update({key:val})
        dense_dropout = DenseDropoutLayers(**dense_params, dropout_p=self.dropout_p)
        return dense_dropout

    def convert_layer(self, layer):
        if isinstance(layer,ConvLayers):
            return self.convert_conv_dropout(layer)
        if isinstance(layer,DenseLayers):
            return self.convert_dense_dropout(layer)
        else:
            return layer

    def convert_torch_sequential(self, torch_sequential):
        converted_branch = []
        for layer in torch_sequential.children():
            converted_branch.append(self.convert_layer(layer))
        return torch.nn.Sequential(*converted_branch)

    def convert_autoencoder(self, autoencoder=Autoencoder):
        encoder = self.convert_torch_sequential(autoencoder.encoder) if isinstance(autoencoder.encoder, torch.nn.Sequential) else self.convert_layer(autoencoder.encoder)
        decoder_mu = self.convert_torch_sequential(autoencoder.decoder_mu) if isinstance(autoencoder.decoder_mu, torch.nn.Sequential) else self.convert_layer(autoencoder.decoder_mu)

        if autoencoder.decoder_sig_enabled:
            decoder_sig = self.convert_torch_sequential(autoencoder.decoder_sig) if isinstance(autoencoder.decoder_sig, torch.nn.Sequential) else self.convert_layer(autoencoder.decoder_sig)
        else:
            decoder_sig = None
        return Autoencoder(encoder=encoder, decoder_mu=decoder_mu, decoder_sig=decoder_sig, homoscedestic_mode=self.homoscedestic_mode)
