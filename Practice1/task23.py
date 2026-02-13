import torch

torch.manual_seed(42)

a = torch.randn(4, 5, 6)

print(a.shape, a.dim(), a.numel())

b = a.permute(2, 0, 1)
print("После permute:", b.shape)

# transpose (меняем две оси за раз)
c = a.transpose(0, 2).transpose(1, 2)         # 6×5×4 → 6×4×5
print("После transpose:", c.shape)

# view vs reshape
flat_view  = a.view(-1, 10)     # -1 = автоматический расчёт
flat_reshape = a.reshape(12, 10)

print("Суммы совпадают?", torch.allclose(a.sum(), flat_view.sum()))
print("view  и reshape идентичны?", torch.equal(flat_view, flat_reshape))