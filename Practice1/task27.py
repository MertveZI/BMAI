import torch

torch.manual_seed(42)

scores = torch.randn(8, 5)
best_idx = ___   # лучший в каждой игре

# gather — берём score лучшего в каждой колонке
best_scores = torch.gather(___, index=best_idx.unsqueeze(0))

print("Лучшие результаты по играм:", best_scores.squeeze())

# index_select — строки по списку
selected_games = torch.index_select(___)
print(selected_games.shape)   # (8,2)

# scatter — проставить -1 всем, кроме лучших
mask = torch.zeros_like(scores)
mask.scatter_(___, index=best_idx.unsqueeze(0), value=1.0)
scores_non_best = scores.clone()
scores_non_best[mask == 0] = -1