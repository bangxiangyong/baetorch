from fastai.layers import SelfAttention, PooledSelfAttention2d
import torch

tst = SelfAttention(16)
x = torch.randn(32, 16, 8, 8)
print(tst(x).shape)

layer_ = PooledSelfAttention2d(16)
print(layer_(x).shape)
