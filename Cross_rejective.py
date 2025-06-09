import time
import numpy as np
import torch
import torch.serialization
torch.serialization.add_safe_globals([torch.nn.parallel.DataParallel])
from torch.cuda.amp import autocast as autocast, GradScaler
from models.simTRM_contrastive import *
from evaluation import openset_eval_contrastive, openset_eval_contrastive_logits
from utils import load_ImageNet200, get_smooth_labels
from mixup import *
from classifier_test import cross_matrix_value
from Encoder_finetune import encoder_finetune
import os
'''This part is the core code of the cross-rejection section'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def Cross_rejective_main(dataset = 'yama', threhods = 0.95, batch_size = 1024, lr = 0.00002, num_classifier_epochs = 11,
                         percentile = 1, label_smoothing_coeff = 0, feature_dim = 128*16, begin_epoch = 6, check_epoch = 5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_root = './data/' + dataset + '/train/'   
    test_R_root = './data/' + dataset + '/test_R/'
    test_S_root = './data/' + dataset + '/test_S/'

    CLASS_NUM_R = len([d for d in os.listdir(test_S_root) if os.path.isdir(os.path.join(test_S_root, d))])
    CLASS_NUM_S = len([d for d in os.listdir(test_R_root) if os.path.isdir(os.path.join(test_R_root, d))])

    print(f"CLASS_NUM_R: {CLASS_NUM_R}")
    print(f"CLASS_NUM_S: {CLASS_NUM_S}")
    TOTAL_CLASS_NUM = CLASS_NUM_S + CLASS_NUM_R

    classid_R = [i for i in range(1, CLASS_NUM_R + 1)]
    classid_S = [i for i in range(CLASS_NUM_R + 1 ,TOTAL_CLASS_NUM + 1)]
    threhods = 0.95
 
    '''Set up where to store the trained model and use this pre-trained model to continue training'''
    model_folder_path = './saved_models/'
    feature_encoder = torch.load(model_folder_path + dataset + '_2000.pt', weights_only = False) 
    feature_encoder = feature_encoder.module
    feature_encoder.to(device) 

    classid_training = feature_encoder.classid_list
    classid_training.sort()
    
    '''Define the dataset loader'''
    train_loader_classifier, train_classes = load_ImageNet200([train_root], category_indexs=classid_training, train=True, batchSize=batch_size, useRandAugment=True)
    validation_loader_R, train_classes_R = load_ImageNet200([train_root], category_indexs=classid_R, train=False, batchSize=batch_size, useRandAugment=False)
    validation_loader_S, train_classes_S = load_ImageNet200([train_root], category_indexs=classid_S, train=False, batchSize=batch_size, useRandAugment=False)
    test_loader_R, test_classes_R = load_ImageNet200([test_R_root], category_indexs = classid_S, train = False, batchSize=batch_size, useRandAugment=False)
    test_loader_S, test_classes_S = load_ImageNet200([test_S_root], category_indexs = classid_R, train = False, batchSize=batch_size, useRandAugment=False)

    scaler_S = GradScaler()
    scaler_R = GradScaler()

    feature_encoder.eval()

    classifier_R = MLPClassifier(classid_list = classid_R, feature_dim=feature_dim) 

    classifier_R.to(device)
    optimizer_classifier_R = torch.optim.Adam(classifier_R.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler_classifier_R = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_classifier_R, T_max=num_classifier_epochs)

    classifier_S = MLPClassifier(classid_list = classid_S, feature_dim=feature_dim) 

    classifier_S.to(device)
    optimizer_classifier_S = torch.optim.Adam(classifier_S.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler_classifier_S = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_classifier_S, T_max=num_classifier_epochs)  
    for classifier_epoch in range(num_classifier_epochs):

        classifier_R.train()
        classifier_S.train()
        time3 = time.time()
        total_classification_loss_R = 0
        total_classification_loss_S = 0
        for i, (images_classifier, labels_classifier) in enumerate(train_loader_classifier):
            length = len(train_loader_classifier)
            images_classifier = images_classifier.to(device)
            labels_np = labels_classifier.numpy()
            labels_classifier = labels_classifier.to(device)
            targets = get_smooth_labels(labels_classifier, classid_training, smoothing_coeff=label_smoothing_coeff) 
            optimizer_classifier_R.zero_grad()
            optimizer_classifier_S.zero_grad()
            '''with autocast() to speed up the training process torch.no_grad is used to speed up reasoning'''
            with autocast():
                with torch.no_grad():
                    features = feature_encoder.get_feature(images_classifier)

                logits_R = classifier_R(features)
                logits_S = classifier_S(features)

                classification_loss_R = classifier_R.get_loss(logits_R, targets, begin_index = 0, end_index = CLASS_NUM_R)
                classification_loss_S = classifier_S.get_loss(logits_S, targets, begin_index = CLASS_NUM_R, end_index = CLASS_NUM_R + CLASS_NUM_S)
                
                if classifier_epoch < begin_epoch:
                    Mix_loss_R = 0
                    Mix_loss_S = 0
                else:
                    # Mix_loss_R = 0
                    # Mix_loss_S = 0
                    Mix_loss_R, _ = classifier_R.get_cross_loss(logits_R = logits_R, logits_S = logits_S, targets = targets, mask_martix_R = mask_martix_R, mask_martix_S = mask_martix_S, begin_index = 0, end_index = CLASS_NUM_R, branch = 'R')
                    Mix_loss_S, _ = classifier_S.get_cross_loss(logits_R = logits_R, logits_S = logits_S, targets = targets, mask_martix_R = mask_martix_R, mask_martix_S = mask_martix_S, begin_index = CLASS_NUM_R, end_index = CLASS_NUM_R + CLASS_NUM_S, branch = 'S')
                batch_loss_R = classification_loss_R + Mix_loss_R
                batch_loss_S = classification_loss_S + Mix_loss_S
            '''Update the magnification scale and update the parameters'''          
            scaler_R.scale(batch_loss_R).backward()
            scaler_R.step(optimizer_classifier_R)
            scaler_R.update()
            optimizer_classifier_R.zero_grad()
            
            scaler_S.scale(batch_loss_S).backward()
            scaler_S.step(optimizer_classifier_S)
            scaler_S.update()
            optimizer_classifier_S.zero_grad()
            total_classification_loss_R += batch_loss_R
            total_classification_loss_S += batch_loss_S

            classifier_R.eval()
            classifier_S.eval()

        if(classifier_epoch % check_epoch == 0 and classifier_epoch != 0):
            mask_martix_R, mask_martix_S, mask_martix = cross_matrix_value(feature_encoder = feature_encoder, test_loader_R = test_loader_R, test_loader_S = test_loader_S,
                                                                           classifier_R = classifier_R, classifier_S = classifier_S, threhods = threhods)

            mask_martix_R = torch.tensor(mask_martix_R)
            torch.save(mask_martix_R, './tensor_result/' + dataset + '_' + str(threhods) +'_tensor_R' + str(classifier_epoch) + '.pt')
            mask_martix_S = torch.tensor(mask_martix_S)
            torch.save(mask_martix_S, './tensor_result/' + dataset + '_' + str(threhods) + '_tensor_S' + str(classifier_epoch) + '.pt')
            mask_martix = torch.tensor(mask_martix)
            torch.save(mask_martix, './tensor_result/' + dataset + '_' + str(threhods) + '_tensor_mix' + str(classifier_epoch) + '.pt')
            encoder_finetune(train_root = train_root, batch_size = 256, feature_encoder = feature_encoder, mask_matrix = mask_martix, TOTAL_CLASS_NUM = TOTAL_CLASS_NUM)
        total_classification_loss_R = total_classification_loss_R/((batch_size)*(length))
        total_classification_loss_S = total_classification_loss_S/((batch_size)*(length))
        torch.cuda.empty_cache()
        '''Renewal learning rate'''
        scheduler_classifier_R.step()
        scheduler_classifier_S.step()

        print('classifier_epoch {}: classification_loss_R = {:.3f}'.format(classifier_epoch, total_classification_loss_R))
        print('classifier_epoch {}: classification_loss_S = {:.3f}'.format(classifier_epoch, total_classification_loss_S))
        time4 = time.time()
        print('time for this epoch: {:.3f} minutes'.format((time4 - time3) / 60.0))

    with autocast():
        start_time = time.time()
        thresholds_R = classifier_R.estimate_threshold_logits(feature_encoder, validation_loader_R,percentile=percentile)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The execution time is: {execution_time} seconds")
        print(thresholds_R) 

    with autocast():
        thresholds_S = classifier_S.estimate_threshold_logits(feature_encoder, validation_loader_S,percentile=percentile)
        print(thresholds_S) 

    torch.save(classifier_R, model_folder_path + dataset + '_classifier_r_5.pt')
    torch.save(classifier_S, model_folder_path + dataset + '_classifier_s_5.pt')
    torch.save(feature_encoder, model_folder_path + dataset + '_encoder_fintune.pt')