from baetorch.baetorch.models_v2.base_autoencoder import AutoencoderModule
from baetorch.baetorch.models_v2.base_layer import TwinOutputModule
import torch


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
                if enc_i == 0:
                    sparse_loss = sparse_loss_new
                else:
                    sparse_loss += sparse_loss_new

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
            # handle sparse activation loss
            if isinstance(block, torch.nn.Sequential):
                sparse_loss_new = torch.abs(x).mean()
                sparse_loss += sparse_loss_new

        return [x, sparse_loss]
