import sys
import os
import Contrastive_pretraining
import Cross_rejective
import calculate_rmse
sys.path.append(os.path.abspath('data_made'))
from data_made import *
from data_made import pretrain_made

if __name__ == '__main__':
    dataset = 'yama'
    image_last_name = "bmp"
    threhods = 0.95
    pretrain_made.made_data(data_name = dataset, image_last_name = image_last_name, 
                            T0 = [[0.9973,0.0011,-9.9636], [-0.0011, 0.9973,-0.5292],[0,0,1]])
    
    Contrastive_pretraining.Contrastive_pretraining_main(dataset = dataset,  batch_size = 1024, lr = 0.0003, num_contrastive_epochs = 50,
                                                     temperature = 0.1, label_smoothing_coeff = 0.0)

    Cross_rejective.Cross_rejective_main(dataset = dataset, threhods = threhods, batch_size = 1024, lr = 0.00004, num_classifier_epochs = 51,
                         percentile = 50, label_smoothing_coeff = 0, feature_dim = 128*16, begin_epoch = 11, check_epoch = 10, image_last_name = image_last_name)

    # calculate_rmse.cal_rmse_main(dataset = dataset, threhods = str(threhods), image_last_name = image_last_name, begin_epoch = 5, end_epoch = 6, indent = 5)