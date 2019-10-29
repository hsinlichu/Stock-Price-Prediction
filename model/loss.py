import torch.nn.functional as F
import torch.nn as nn
import torch

def MSELoss(output, target):
    output = torch.squeeze(output)
    target = torch.squeeze(target)
    return nn.MSELoss()(output, target)
