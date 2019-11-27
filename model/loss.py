import torch.nn.functional as F
import torch.nn as nn
import torch

def MSELoss(output, target):
    output = torch.squeeze(output)
    target = torch.squeeze(target)
    return F.mse_loss(output, target)

def VAEandMSELoss(out, target):
    recon_x, x, mu, logvar, output = out

    output = torch.squeeze(output)
    target = torch.squeeze(target)

    '''
    print("=====")
    print(recon_x.size(), x.size())
    print(recon_x[0][0], x[0][0])
    '''
    recon_loss = F.mse_loss(recon_x, x) # i don't know if it is correct
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    MSE = F.mse_loss(output, target)
    #print(recon_loss, KLD, MSE)

    return recon_loss + KLD + MSE

