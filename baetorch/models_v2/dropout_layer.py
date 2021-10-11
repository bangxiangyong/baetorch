import torch
from torch.nn import Parameter


class CustomDropout(torch.nn.Module):
    def __init__(self, drop_p=0.1, n_samples=100):
        super(CustomDropout, self).__init__()
        self.dropout_rate = drop_p
        self.n_samples = n_samples

        self.mask_ready = False

    def init_mask(self, x):
        inp_size = x.shape[1:]

        samples = torch.distributions.bernoulli.Bernoulli(1 - self.dropout_rate).sample(
            (self.n_samples, *inp_size)
        )
        self.mask = Parameter(torch.FloatTensor(samples), requires_grad=False)

        # check if input is on Cuda
        # set the mask to cuda
        if x.is_cuda:
            self.cuda()

        self.counter_i = 0

    def forward(self, x):
        # init mask dynamically based on input
        if not self.mask_ready:
            self.init_mask(x)
            self.mask_ready = True

        # actually compute mask on input
        output = x * self.mask[self.counter_i]

        # handle counter
        self.counter_i += 1
        if self.counter_i >= self.n_samples:
            self.counter_i = 0
        return output
