from util.metrics import *
import torch

y = torch.Tensor([1.0, 0.0, 0.0, 1.0]).view(-1,1)
y_hat = torch.Tensor([1.0, 0.0, 1.0, 1.0])

print(binary_accuracy(y, y_hat))
