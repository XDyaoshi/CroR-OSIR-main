import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ABN import MultiBatchNorm
import torchvision.models as tmodel
from models.swin_trm import SwinTransformerCustom

trm_model = SwinTransformerCustom()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('MultiBatchNorm') != -1:
        m.bns[0].weight.data.normal_(1.0, 0.02)
        m.bns[0].bias.data.fill_(0)
        m.bns[1].weight.data.normal_(1.0, 0.02)
        m.bns[1].bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class simTRM_contrastive(nn.Module):
    def __init__(self, classid_list, num_ABN=5, head='mlp', feature_dim=128):
        super(self.__class__, self).__init__()
        self.num_classes = len(classid_list)
        self.feature_dim = feature_dim
        self.logit_dim = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classid_list = classid_list
        self.trm  = trm_model

        self.maxpool = nn.AdaptiveMaxPool2d((4, 4))
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        if head == 'linear':   #For classification learning
            self.head = nn.utils.spectral_norm(nn.Linear(self.feature_dim, self.logit_dim))
        elif head == 'mlp':    #For contrastive learning
            self.head = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(self.feature_dim*16, self.feature_dim*16)), 
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Linear(self.feature_dim*16, self.logit_dim))
            )

        # self.apply(weights_init) #Initializing the weights
        self.to(self.device)


    def get_feature(self, x, bn_label=None):
        if bn_label is None:
            bn_label = 0 * torch.ones(x.shape[0], dtype=torch.long).cuda()
        x = self.trm(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        return x
    
    

    def get_output(self, features):
        logits = F.normalize(self.head(features), dim=1)
        return logits

    def forward(self, x):
        features = self.get_feature(x)
        logits = self.get_output(features)
        return logits

    def freeze_weight(self): #If the gradient of a parameter is frozen and not allowed to be calculated, it will not be updated
        for param in self.parameters():
            param.requires_grad = False


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, classid_list, feature_dim=128):
        super(LinearClassifier, self).__init__()
        self.num_classes = len(classid_list)
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classid_list = classid_list
        self.fc = nn.utils.spectral_norm(nn.Linear(feature_dim, self.num_classes))

    def forward(self, features):
        logits = self.fc(features)
        return logits

    def get_prob(self, logits, temperature=1):
        probs = torch.softmax(logits / temperature, dim=1)
        return probs 
    
    def get_logprob(self, logits, temperature=1):
        probs = torch.log_softmax(logits / temperature, dim=1)
        return probs #The value of the probability distribution

    def get_loss(self, logits, targets, temperature=1, begin_index = 0, end_index = 4499):
        log_probs = torch.log_softmax(logits / temperature, dim=1)
        targets = targets[:, begin_index:end_index]
        '''After this step, the labels are filtered so that only a few categories corresponding to the classifier categories are retained'''
        loss = -torch.sum(log_probs * targets)
        
        return loss

    def get_mix_loss(self, logits_R, logits_S, targets, temperature=1, begin_index=0, end_index=4499, branch='R'):
        # Make sure all data is on the GPU
        device = logits_R.device
        targets = targets.to(device)
        logits_R = logits_R.to(device)
        logits_S = logits_S.to(device)
    
        target_indices = torch.where(targets == 1)[1]  # Counts the labels of all samples in a batch

        if begin_index == 0:
            class_R_index = torch.nonzero(target_indices < end_index, as_tuple=True)[0]
            class_S_index = torch.nonzero(target_indices >= end_index, as_tuple=True)[0]
        else:
            class_R_index = torch.nonzero(target_indices < begin_index, as_tuple=True)[0]
            class_S_index = torch.nonzero(target_indices >= begin_index, as_tuple=True)[0]
        # Determine which samples belong to the class in R and which belong to the class in S
        if branch == 'R':
            pesudo_label_R = torch.argmax(logits_S[class_R_index], dim=1) + end_index + 1
            # pesudo_label_R explicitly corresponds to the category rather than the index, and the category R is predicted using the classification head of the S branch
            probs_S = self.get_prob(logits=logits_S, temperature=temperature)
            probs = probs_S[class_R_index].max(dim=1)[0]

            pesudo_label = torch.stack((target_indices[class_R_index], pesudo_label_R), dim=1)
            # The values of this tag are key-value pairs i.e., [R tag (pseudo-tag)<-S tag]
            mask_martix = torch.zeros((targets.shape[0], len(self.classid_list)), device=device)
            # Pseudo labels are generated on the R branch
            for idx in class_S_index:
                mask = (pesudo_label[:, 1] - 1) == target_indices[idx]
                # Find the real label that the pseudo-label is in, and assign it the value at the corresponding position of the pseudo-label
                mask_martix[idx, pesudo_label[mask][:, 0]] = (probs[mask]>=0.1).float()
        
            probs_R = self.get_prob(logits=logits_R, temperature=temperature)
        else:
            pesudo_label_S = torch.argmax(logits_R[class_S_index], dim=1) + 1
            probs_R = self.get_prob(logits=logits_R, temperature=temperature)
            probs = probs_R[class_S_index].max(dim=1)[0]

            pesudo_label = torch.stack((target_indices[class_S_index], pesudo_label_S), dim=1)
            mask_martix = torch.zeros((targets.shape[0], len(self.classid_list)), device=device)
            # The process of generating pseudo-labels on the S branch
            for idx in class_R_index:
                mask = (pesudo_label[:, 1] - 1) == target_indices[idx]
                mask_martix[idx, pesudo_label[mask][:, 0] - begin_index] = (probs[mask]>=0.1).float()

            probs_S = self.get_prob(logits=logits_S, temperature=temperature)

        mix_loss = -torch.sum(probs_R * mask_martix) if branch == 'R' else -torch.sum(probs_S * mask_martix)
        return mix_loss, pesudo_label


    def get_cross_loss(self, logits_R, logits_S, targets, mask_martix_R, mask_martix_S, temperature=1, begin_index=0, end_index=4499, branch='R'):
        # Make sure all data is on the GPU
        device = logits_R.device
        targets = targets.to(device)
        logits_R = logits_R.to(device)
        logits_S = logits_S.to(device)

        mask_martix_R = mask_martix_R.to(device)
        mask_martix_S = mask_martix_S.to(device)

        target_indices = torch.where(targets == 1)[1]  # Counts the labels of all samples in a batch

        if begin_index == 0:
            class_R_index = torch.nonzero(target_indices < end_index, as_tuple=True)[0]
            class_S_index = torch.nonzero(target_indices >= end_index, as_tuple=True)[0]
        else:
            class_R_index = torch.nonzero(target_indices < begin_index, as_tuple=True)[0]
            class_S_index = torch.nonzero(target_indices >= begin_index, as_tuple=True)[0]
        # Determine which samples belong to the class in R and which belong to the class in S
        if branch == 'R':

            pesudo_label = mask_martix_R
            probs = pesudo_label[:,2]
            mask_martix = torch.zeros((targets.shape[0], len(self.classid_list)), device=device)

            # The process of generating pseudo-labels on the R branch
            for idx in class_S_index:
                mask = (pesudo_label[:, 1] - 1) == target_indices[idx]
                # Find the real label that the pseudo-label is in, and assign it the value at the corresponding position of the pseudo-label
                mask_martix[idx, pesudo_label[mask][:, 0].int() - 1] = (probs[mask] >= 0.0).float()
        
            probs_R = self.get_logprob(logits=logits_R, temperature=temperature)
        else:
           
            pesudo_label = mask_martix_S
            pesudo_label[:, [0, 1]] = pesudo_label[:, [1, 0]]
            probs = pesudo_label[:,2]
            mask_martix = torch.zeros((targets.shape[0], len(self.classid_list)), device=device)
            # Pseudo labels are generated on the S branch
            for idx in class_R_index:
                mask = (pesudo_label[:, 1] - 1) == target_indices[idx]
                mask_martix[idx, pesudo_label[mask][:, 0].int() - begin_index - 1] = (probs[mask] >= 0.0).float()

            probs_S = self.get_logprob(logits=logits_S, temperature=temperature)

        mix_loss = -torch.sum(probs_R * mask_martix) if branch == 'R' else -torch.sum(probs_S * mask_martix)
        # mix_loss = torch.abs(mix_loss)       
        return mix_loss, pesudo_label

    def estimate_threshold(self, probs, labels, percentile=5):
        self.classwise_thresholds = []
        classwise_probs = []
        for i in range(self.num_classes):
            classwise_probs.append([]) 

        for i, val in enumerate(probs):
            if self.classid_list.count(labels[i]) > 0:
                id_index = self.classid_list.index(labels[i])
                maxProb = np.max(probs[i])
                if probs[i, id_index] == maxProb:
                    classwise_probs[id_index].append(probs[i, id_index])

        for val in classwise_probs:
            if len(val) == 0:
                self.classwise_thresholds.append(0)
            else:
                threshold = np.percentile(val, percentile)
                self.classwise_thresholds.append(threshold)

        return self.classwise_thresholds

    def estimate_threshold_logits(self, feature_encoder, validation_loader, percentile=5):
        self.eval()
        self.classwise_thresholds = []
        classwise_logits = []
        classwise_logits_wrong = []
        for i in range(self.num_classes):
            classwise_logits.append([]) 
            classwise_logits_wrong.append([])

        for i, (images, labels) in enumerate(validation_loader):
        # The threshold is calculated using the verification set
            images = images.to(self.device)
            with torch.no_grad():
                features = feature_encoder.get_feature(images)
                logits = self.forward(features)
                maxLogit, maxIndexes = torch.max(logits, 1) #Take the label with the highest probability as the predicted value
                
            for j, label in enumerate(labels):   
                id_index = self.classid_list.index(label)
                
                if maxIndexes[j] == id_index:
                    classwise_logits[id_index].append(logits[j, id_index].item()) #If the samples are split correctly, the logits corresponding to the split samples are added to the candidate list
                else:
                    classwise_logits_wrong[maxIndexes[j]].append(-1)
                    continue

        max_len =  0
        for val in classwise_logits_wrong:
            if(max_len < len(val)):
                max_len = len(val)
    
        for i in range(len(classwise_logits)):
            val = classwise_logits[i]
            if len(val) == 0:
                self.classwise_thresholds.append(1000)
                # That is, no samples are classified into this category, which indicates that this kind of samples are OOD data
            else:
                # lamda = len(classwise_logits_wrong[i])/(max_len + 1)
                # lamda = int(lamda)
                # percentile = percentile + (100 - percentile) * lamda
                # percentile = int(percentile)
                percentile = 5

                threshold = np.percentile(val, percentile) #Calculating percentiles
                self.classwise_thresholds.append(threshold)
        return self.classwise_thresholds


    def predict_logits(self, features):
        logits = self.forward(features)

        maxLogits, maxIndexes = torch.max(logits, 1)

        prediction = torch.zeros([maxIndexes.shape[0]], requires_grad=False).to(self.device)

        for i in range(maxIndexes.shape[0]):
            prediction[i] = self.classid_list[maxIndexes[i]]
            if maxLogits[i] <= self.classwise_thresholds[maxIndexes[i]]:
                prediction[i] = -1

        return prediction.long(), logits.detach().cpu().numpy()
    
    def predict_logits_pre(self, features, threhods):
        logits = self.forward(features)
        probs = self.get_logprob(logits = logits,temperature = 1)
        maxProbs, maxProbIndexes = torch.max(probs, 1)
        maxLogits, maxIndexes = torch.max(logits, 1)
        prediction = torch.zeros([maxIndexes.shape[0]], requires_grad=False).to(self.device)

        for i in range(maxIndexes.shape[0]):
            prediction[i] = self.classid_list[maxIndexes[i]]
            if maxLogits[i] <= self.classwise_thresholds[maxIndexes[i]] * threhods:
            # if maxProbs[i] <= 0.75:
                prediction[i] = -1

        return prediction.long(), logits.detach().cpu().numpy(), maxProbs.detach().cpu().numpy()

    def predict_closed(self, features):
        outs = self.forward(features)
        probs = torch.sigmoid(outs)

        maxProb, maxIndexes = torch.max(probs, 1)
        prediction = torch.zeros([maxIndexes.shape[0]], requires_grad=False).to(self.device)

        for i in range(maxIndexes.shape[0]):
            prediction[i] = self.classid_list[maxIndexes[i]]

        return prediction.long()


class MLPClassifier(LinearClassifier):
    """MLP classifier"""
    def __init__(self, classid_list, feature_dim=128):
        super(MLPClassifier, self).__init__(classid_list, feature_dim)

        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.feature_dim, self.feature_dim)),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Linear(self.feature_dim, self.num_classes))
        )
