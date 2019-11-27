import torch
import torch.nn.functional as F


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def difference(output, target):
    with torch.no_grad():
        delta = target - output
        average = torch.mean(delta, 0)
        return average.item()

def difference_vae(out, target):
    recon_x, x, mu, logvar, output = out
    with torch.no_grad():
        delta = target - output
        average = torch.mean(delta, 0)
        return average.item()

def mse_vae(out, target):
    recon_x, x, mu, logvar, output = out
    with torch.no_grad():
        output = torch.squeeze(output)
        target = torch.squeeze(target)
        return F.mse_loss(output, target)




