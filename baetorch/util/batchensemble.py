import math
import torch
import torch.nn as nn


class Ensemble_FC(nn.Module):
    def __init__(self, in_features, out_features, first_layer, num_models, bias=True):
        super(Ensemble_FC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        # self.alpha = torch.Tensor(num_models, in_features).cuda()
        # self.gamma = torch.Tensor(num_models, out_features).cuda()
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_features))
        self.gamma = nn.Parameter(torch.Tensor(num_models, out_features))
        self.num_models = num_models
        if bias:
            # self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.first_layer = first_layer

    def reset_parameters(self):
        # nn.init.constant_(self.alpha, 1.0)
        # nn.init.constant_(self.gamma, 1.0)
        nn.init.normal_(self.alpha, mean=1.0, std=0.1)
        nn.init.normal_(self.gamma, mean=1.0, std=0.1)
        # nn.init.normal_(self.alpha, mean=1., std=1)
        # nn.init.normal_(self.gamma, mean=1., std=1)
        # alpha_coeff = torch.randint_like(self.alpha, low=0, high=2)
        # alpha_coeff.mul_(2).add_(-1)
        # gamma_coeff = torch.randint_like(self.gamma, low=0, high=2)
        # gamma_coeff.mul_(2).add_(-1)
        # with torch.no_grad():
        #    self.alpha *= alpha_coeff
        #    self.gamma *= gamma_coeff
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def update_indices(self, indices):
        self.indices = indices

    def forward(self, x):
        if self.training:
            curr_bs = x.size(0)
            makeup_bs = self.num_models - curr_bs
            if makeup_bs > 0:
                indices = torch.randint(
                    high=self.num_models, size=(curr_bs,), device=self.alpha.device
                )
                alpha = torch.index_select(self.alpha, 0, indices)
                gamma = torch.index_select(self.gamma, 0, indices)
                bias = torch.index_select(self.bias, 0, indices)
                result = self.fc(x * alpha) * gamma + bias
            elif makeup_bs < 0:
                indices = torch.randint(
                    high=self.num_models, size=(curr_bs,), device=self.alpha.device
                )
                alpha = torch.index_select(self.alpha, 0, indices)
                gamma = torch.index_select(self.gamma, 0, indices)
                bias = torch.index_select(self.bias, 0, indices)
                result = self.fc(x * alpha) * gamma + bias
            else:
                result = self.fc(x * self.alpha) * self.gamma + self.bias
            return result[:curr_bs]
        else:
            if self.first_layer:
                # Repeated pattern: [[A,B,C],[A,B,C]]
                x = torch.cat([x for i in range(self.num_models)], dim=0)
            # Repeated pattern: [[A,A],[B,B],[C,C]]
            batch_size = int(x.size(0) / self.num_models)
            alpha = torch.cat([self.alpha for i in range(batch_size)], dim=1).view(
                [-1, self.in_features]
            )
            gamma = torch.cat([self.gamma for i in range(batch_size)], dim=1).view(
                [-1, self.out_features]
            )
            bias = torch.cat([self.bias for i in range(batch_size)], dim=1).view(
                [-1, self.out_features]
            )
            result = self.fc(x * alpha) * gamma + bias
            return result
