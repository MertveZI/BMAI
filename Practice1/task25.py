import torch

torch.manual_seed(42)

A = torch.randn(32, 10)
B = torch.randn(10, 8)
C = torch.randn(32, 1)
D = torch.randn(8)

# Матричное умножение
out1 = ___

#  + bias (broadcast)
out2 = ___

#  × веса на всю партию (broadcast)
out3 = ___

#  einsum-вариант (всё сразу)
out_einsum = torch.einsum(___) + ___

print("Результаты совпадают?", torch.allclose(out3, out_einsum, atol=1e-6))