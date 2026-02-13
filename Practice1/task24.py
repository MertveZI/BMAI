import torch

torch.manual_seed(42)

A = torch.randn(32, 10)
B = torch.randn(10, 8)
C = torch.randn(32, 1)
D = torch.randn(8)

# Матричное умножение
out1 = A @ B

# + bias (broadcast)
out2 = out1 + C

# × веса на всю партию (broadcast)
out3 = out2 * D

# einsum-вариант (всё сразу)
out_einsum = torch.einsum('bi,ij,j->bj', A, B, D) + C * D

print("Результаты совпадают?", torch.allclose(out3, out_einsum, atol=1e-6))