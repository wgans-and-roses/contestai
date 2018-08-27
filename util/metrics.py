import torch


def binary_accuracy(y, y_hat):
    y = y.view(-1); y_hat = y_hat.view(-1)  # this is to avoid problems if one vector is a column and the other is a row
    acc = torch.sum(y == y_hat).float()/y.size()[0]
    return acc
