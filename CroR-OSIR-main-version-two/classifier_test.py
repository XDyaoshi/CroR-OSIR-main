import time
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast, GradScaler
from models.simTRM_contrastive import *
from evaluation import openset_eval_contrastive, openset_eval_contrastive_logits
from utils import load_ImageNet200, get_smooth_labels
from mixup import *
import os            
'''The execution of the cross-reject module is defined'''
def cross_matrix_value(feature_encoder, test_loader_R, test_loader_S, classifier_R, classifier_S, threhods = 0.85):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    percentile = 99 
    feature_encoder.to(device) 

    classid_training = feature_encoder.classid_list
    classid_training.sort()
    
    feature_encoder.eval()
    classifier_R.to(device)
    classifier_S.to(device)

    classifier_R.eval()
    classifier_S.eval()
    thresholds_R = classifier_R.estimate_threshold_logits(feature_encoder, test_loader_S, percentile = percentile)
    thresholds_S = classifier_S.estimate_threshold_logits(feature_encoder, test_loader_R, percentile = percentile)
    
    mask_martix_R = []
    mask_martix_S = []
    score_R = []
    score_S = []
    for images, labels in test_loader_R:
        images = images.to(device) 
        # Images
        labels = labels.to(device)
        # Labels
        features = feature_encoder.get_feature(images)
        
        prediction, logits, probs = classifier_R.predict_logits_pre(features, threhods = threhods)
        # prediction
        for i in range(len(prediction)):
            if (prediction[i].item() != -1):
                mask_martix_R.append([int(prediction[i].item()), int(labels[i].item()), probs[i]])
                score_R.append([probs[i]])

    
    for images, labels in test_loader_S:
        images = images.to(device) 
        # Images
        labels = labels.to(device)
        # Labels
        features = feature_encoder.get_feature(images)
        
        prediction, logits, probs = classifier_S.predict_logits_pre(features, threhods = threhods)
        # prediction
        for i in range(len(prediction)):
            if (prediction[i].item() != -1):
                mask_martix_S.append([int(labels[i].item()), int(prediction[i].item()), probs[i]])
                score_S.append([probs[i]])


    mask_martix_R = np.array(mask_martix_R)
    mask_martix_S = np.array(mask_martix_S)
    mask_martix_R = mask_martix_R[mask_martix_R[:, 0].argsort()]
    mask_martix_S = mask_martix_S[mask_martix_S[:, 0].argsort()]
    i, j = 0, 0
    result = []# Since both arrays are ordered, you can use a double pointer traversal
    while i < len(mask_martix_R) and j < len(mask_martix_S):
        if np.array_equal(mask_martix_R[i, :2], mask_martix_S[j, :2]):
        # Compare the values in the third column, keeping the larger ones
            if mask_martix_R[i, 2] > mask_martix_S[j, 2]:
                result.append(mask_martix_R[i])
            else:
                result.append(mask_martix_S[j])
            i += 1
            j += 1
        elif (mask_martix_R[i, 0] < mask_martix_S[j, 0]) or (mask_martix_R[i, 0] == mask_martix_S[j, 0] and mask_martix_R[i, 1] < mask_martix_S[j, 1]):
            i += 1
        else:
            j += 1


    # result_array = np.array(result)
    unique_rows = {}
    for row in result:
        key = tuple(row[:2])
        if key not in unique_rows or row[2] > unique_rows[key][2]:
            unique_rows[key] = row


    mask_array = np.array(list(unique_rows.values()))

    return mask_martix_R, mask_martix_S, mask_array