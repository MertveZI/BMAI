import torch

torch.manual_seed(42)

v1 = torch.randn(128)
v2 = torch.randn(128)

A = torch.randn(64, 32)
B = torch.randn(32, 16)

batch_feat = torch.randn(16, 128)
weight = torch.randn(128, 64)

#  dot product
dot = ___

# matmul
matmul = ___

# batch matmul
batch_out = ___

# outer product
outer = ___

# trace
M = torch.randn(30, 30)
tr = ___

# 6. global average по channel,h,w
images = torch.randn(8, 3, 224, 224)
global_sum = ___