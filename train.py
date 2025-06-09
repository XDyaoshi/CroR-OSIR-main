import Contrastive_pretraining
import Cross_rejective
import calculate_rmse

if __name__ == '__main__':
    dataset = 'yellowA'
    Contrastive_pretraining.Contrastive_pretraining_main(dataset = dataset,  batch_size = 1024, lr = 0.0003, num_contrastive_epochs = 51,
                                                     temperature = 0.1, label_smoothing_coeff = 0.1)

    Cross_rejective.Cross_rejective_main(dataset = dataset, threhods = 0.95, batch_size = 1024, lr = 0.00002, num_classifier_epochs = 51,
                         percentile = 95, label_smoothing_coeff = 0, feature_dim = 128*16, begin_epoch = 6, check_epoch = 5)

    calculate_rmse.cal_rmse_main(dataset = dataset, threhods = '0.95', image_last_name = "bmp")