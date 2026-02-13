import torch

torch.manual_seed(42)

x1 = torch.randn(4, 6)
x2 = torch.randn(4, 6)
x3 = torch.randn(4, 6)

# concat по разным осям
cat_dim0 = torch.cat((x1, x2, x3), (12,6))
cat_dim1 = torch.cat((x1, x2, x3), (4,18))

# stack → новое измерение
stacked = torch.stack((x1, x2, x3), (3,4,6))     # (3,4,6)

# разрезание stacked обратно
x1r, x2r, x3r = torch.unbind(stacked, dim=0)

# chunk — делим по строкам
chunks = torch.chunk(cat_dim0, chunks=3, dim=[4,6])
print([ch.shape for ch in chunks])