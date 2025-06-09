import time
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast, GradScaler
from models.simTRM_contrastive import *
from models.ContrasiveLoss_SoftLabel import *
from evaluation import openset_eval_contrastive_logits
from utils import load_ImageNet200, load_ImageNet200_contrastive, get_smooth_labels
from mixup import *
import os
'''The program is the first important part, the SupCon part of the pre-training'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def Contrastive_pretraining_main(dataset = 'yama', batch_size = 1024, lr = 0.0003, num_contrastive_epochs = 11,
                                 temperature = 0.1, label_smoothing_coeff = 0.1):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_root = './data/' + dataset + '/train/'  
    test_R_root = './data/' + dataset + '/test_R/'
    test_S_root = './data/' + dataset + '/test_S/'
    
    CLASS_NUM_R = len([d for d in os.listdir(test_S_root) if os.path.isdir(os.path.join(test_S_root, d))])
    CLASS_NUM_S = len([d for d in os.listdir(test_R_root) if os.path.isdir(os.path.join(test_R_root, d))])
    TOTAL_CLASS_NUM = CLASS_NUM_R + CLASS_NUM_S
    classid_all = [i for i in range(0, TOTAL_CLASS_NUM)]

    classid_training = list(range(1, TOTAL_CLASS_NUM + 1))
    classid_training.sort()
   
    
    '''Set parameters, where to put the trained model'''
    model_folder_path = './saved_models/'
    start_time = time.time()
    print("==> Training Class: ", classid_training)

    train_loader_contrastive, train_classes = load_ImageNet200_contrastive([train_root], category_indexs=classid_training, batchSize=batch_size)
    best_epoch = -1
    best_auc = 0

    feature_encoder = simTRM_contrastive(classid_list=classid_training, head='mlp') 
    feature_encoder = nn.DataParallel(feature_encoder)

    criterion = SupConLoss(temperature=temperature, base_temperature=temperature)
    criterion.to(device)

    optimizer = torch.optim.Adam(feature_encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = lr * 1e-3) #利用余弦退火再次包装

    scaler = GradScaler()
    epoches = []
    losses = []
    for epoch in range(1, num_contrastive_epochs+1):

        feature_encoder.train() 
        time1 = time.time()
        total_loss = 0  
        for i, (images, labels) in enumerate(train_loader_contrastive): 
            length = len(train_loader_contrastive)
            targets = get_smooth_labels(labels, classid_training, label_smoothing_coeff) 
            # images_mixup, targets_mixup, targets_a, targets_b, lam = mixup_data_contrastive(images, targets, alpha=1,use_cuda=False)
            
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
        print('epoch {}: contrastive_loss = {:.3f},  '.format(epoch, total_loss))
        time2 = time.time()
        scheduler.step()
        print('time for this epoch: {:.3f} minutes'.format((time2 - time1) / 60.0))
        if (epoch%50==0):
            epoches.append(epoch)
            losses.append(total_loss)
    print(epoches)
    print(losses)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
    torch.save(feature_encoder, model_folder_path + dataset + '_2000.pt')