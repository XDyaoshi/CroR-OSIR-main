import numpy as np
import torch
'''A means of data enhancement, deprecated'''
def mixup_data_random(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    y_mix = lam * y_a + (1 - lam) * y_b
    return mixed_x, y_mix, y_a, y_b, lam

def mixup_data_contrastive(x, y, alpha=1.0, use_cuda=True):

    if alpha > 0.:
        lam = np.random.beta(alpha, alpha) #Generate random numbers that follow a beta distribution
    else:
        lam = 1.
    batch_size = x[0].size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x=[]
    mixed_x.append(lam * x[0] + (1 - lam) * x[0][index,:]) #In this batch, the two augmented views are mixed separately with the rest of the batch
    mixed_x.append(lam * x[1] + (1 - lam) * x[1][index,:])
    y_a, y_b = y, y[index]
    y_mix = lam * y_a + (1 - lam) * y_b #It doesn't make sense to mix the labels in the same way
    return mixed_x, y_mix, y_a, y_b, lam

