import torch
from torchmetrics import JaccardIndex

target = torch.randint(0, 3, (3, 8, 8))
pred = torch.tensor(target)
pred[2, 3:5, 3:5] = 1 - pred[2, 3:5, 3:5]
jaccard = JaccardIndex(num_classes=3, task="multiclass")
score = jaccard(pred, target)
print(target)
print('\n')
print(pred)
print(score)