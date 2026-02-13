import torch

torch.manual_seed(42)

v1 = torch.randn(128)
v2 = torch.randn(128)

A = torch.randn(64, 32)
B = torch.randn(32, 16)

batch_feat = torch.randn(16, 128)
weight = torch.randn(128, 64)

# dot product
dot = torch.einsum('i,i->', v1, v2)

# matmul
matmul = torch.einsum('ij,jk->ik', A, B)

# batch matmul: (b,i) @ (i,j) → (b,j)
batch_out = torch.einsum('bi,ij->bj', batch_feat, weight)

# outer product
outer = torch.einsum('i,j->ij', v1, v2)

# trace
M = torch.randn(30, 30)
tr = torch.einsum('ii->', M)

# 6. global average по channel,h,w
images = torch.randn(8, 3, 224, 224)
global_sum = torch.einsum('bchw->b', images)
