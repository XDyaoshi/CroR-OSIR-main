import time
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast, GradScaler
from models.simTRM_contrastive import *
from models.ContrasiveLoss_SoftLabel import *
from evaluation import openset_eval_contrastive_logits
from utils import *
from mixup import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
'''This part of the code is also very important as it defines the process of fine-tuning the SupCon module.'''
def encoder_finetune(train_root, batch_size, feature_encoder, mask_matrix, TOTAL_CLASS_NUM):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    classid_training = list(range(1, TOTAL_CLASS_NUM + 1))
    classid_training.sort()

    '''This part is setting the parameters'''

    lr = 0.001
    temperature = 0.1
    label_smoothing_coeff = 0
    
    print("==> Training Class: ", classid_training)

    train_loader_contrastive, _ = load_ImageNet200_contrastive([train_root], category_indexs=classid_training, batchSize=batch_size)
 

    criterion = SupConLoss(temperature=temperature, base_temperature=temperature)
    criterion.to(device)

    optimizer = torch.optim.Adam(feature_encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = lr * 1e-3) #利用余弦退火再次包装

    scaler = GradScaler()

    feature_encoder.train() 
    time1 = time.time()
    total_loss = 0  
    for i, (images, labels) in enumerate(train_loader_contrastive): 
        length = len(train_loader_contrastive)
        targets = get_mixed_smooth_labels(labels, classid_training, mask_matrix, label_smoothing_coeff) 

        images = torch.cat([images[0], images[1]], dim=0) 
        images = images.to(device)
        targets = targets.to(device)

        bsz = targets.shape[0]
        optimizer.zero_grad()
        with autocast():
            logits = feature_encoder(images)
            logits1, logits2 = torch.split(logits, [bsz, bsz], dim=0)
            logits = torch.cat([logits1.unsqueeze(1), logits2.unsqueeze(1)], dim=1)

            logits_combine = torch.cat([logits], dim=0)
            targets_combine = torch.cat([targets], dim=0)
            loss = criterion(logits_combine, targets_combine)
                
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    total_loss += loss
    total_loss = total_loss/(4*batch_size*length)
    time2 = time.time()
    scheduler.step()
    print('time for this fintune: {:.3f} minutes'.format((time2 - time1) / 60.0))
    print(total_loss)
