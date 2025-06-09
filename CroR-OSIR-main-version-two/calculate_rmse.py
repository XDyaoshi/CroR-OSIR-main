import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Rmse_tool
'''This program is used to output the main metrics, R_mse and R_mseloo'''
def cal_rmse_main(dataset = 'yama', threhods = '0.95', image_last_name = "bmp", begin_epoch = 5, end_epoch = 51, indent = 5):

    len_tensor_R = []
    len_tensor_S = []
    
    imgpath_r = './data/' + dataset + '/reference.' + image_last_name
    imgpath_s = './data/' + dataset + '/sensed.' + image_last_name
    photo_r = cv2.imread(imgpath_r, flags = 1)
    photo_s = cv2.imread(imgpath_s, flags = 1)
 
    r_root = './data/' + dataset + '/ref_data.csv'
    s_root = './data/' + dataset + '/sen_n_data.csv'

    r_point = pd.read_csv(r_root, header = None, dtype = int)
    s_point = pd.read_csv(s_root, header = None, dtype = int)
    r_point = np.array(r_point)
    r_point = r_point[1:]
    s_point = np.array(s_point)
    s_point_unchange = np.array(s_point)

    for it in s_point:
        it[0] = it[0] + photo_r.shape[1]

    rmse_list = []
    rmse_loo_list = []
    for epoch in range(begin_epoch, end_epoch, indent):
        tensor_R = torch.load('./tensor_result/' + dataset + '_' + threhods + '_' + 'tensor_R' + str(epoch) +'.pt')
        tensor_S = torch.load('./tensor_result/' + dataset + '_' + threhods + '_' + 'tensor_S' + str(epoch) +'.pt')
        tensor_mix = torch.load('./tensor_result/' + dataset + '_' + threhods + '_' + 'tensor_mix' + str(epoch) +'.pt')
    
        print(len(tensor_mix))
        len_tensor_R.append(len(tensor_R))
        len_tensor_S.append(len(tensor_S))

        combine_img = np.hstack((photo_r, photo_s))
    ref_list = []
    sen_list = []
    sen_list_unchange = []

    for item in tensor_mix:
        ref_point = r_point[item[0].int() - 1]
        sen_point = s_point[item[1].int() - len(r_point) - 1]
        ref_list.append(r_point[item[0].int() - 1].tolist())
        sen_list.append(s_point[item[1].int() - len(r_point) - 1].tolist())
        sen_list_unchange.append(s_point_unchange[item[1].int() - len(r_point) - 1].tolist())
        combine_img = cv2.line(combine_img, ref_point, sen_point, color = (0, 255, 0), thickness = 2)
        cv2.circle(combine_img, sen_point, 5, (255, 0, 0), -1)
        cv2.circle(combine_img, ref_point, 5, (0 , 0, 255), -1)
    save_ref = np.array(ref_list)
    save_sen = np.array(sen_list_unchange)
    save_ref = pd.DataFrame(save_ref)
    save_sen = pd.DataFrame(save_sen)
    save_ref.to_csv('./match_results/' + dataset + '_' + threhods + str(epoch) + '_matched_ref_points.csv', header = None)
    save_sen.to_csv('./match_results/' + dataset + '_' + threhods + str(epoch) + '_matched_sen_points.csv', header = None)
    ref_array = np.array(ref_list, dtype = float)
    sen_array = np.array(sen_list, dtype = float)
    tool_martix = np.ones((len(sen_list),1),dtype=float)
    sen_array = np.concatenate((sen_array, tool_martix), axis=1, dtype = float)
    ref_array = np.concatenate((ref_array, tool_martix), axis=1, dtype = float)
    matrix_1 = np.dot(sen_array.T,sen_array)
    matrix_1 = np.linalg.inv(matrix_1)
    matrix_2 = np.dot(sen_array.T,ref_array)
    matrix_trans = np.dot(matrix_1,matrix_2)

    rmse = Rmse_tool.rmse(refpoint = ref_array, senpoint = sen_array, matrix = matrix_trans)

    rmse_list.append(rmse)
    r_loo = 0
    for i in range(len(ref_array)):
        row_to_delete = i
        new_ref = np.delete(ref_array,row_to_delete,axis = 0)
        new_sen = np.delete(sen_array,row_to_delete,axis = 0)
        matrix_1 = np.dot(new_sen.T,new_sen)
        matrix_1 = np.linalg.inv(matrix_1)
        matrix_2 = np.dot(new_sen.T,new_ref)
        matrix_trans = np.dot(matrix_1,matrix_2)
        # print(matrix_trans)
        r_loo = r_loo + Rmse_tool.rmse(refpoint = ref_array, senpoint = sen_array, matrix = matrix_trans)
    # print(r_loo/len(ref_array))
    R_loo = r_loo/len(ref_array)
    rmse_loo_list.append(r_loo/len(ref_array))
    save_rmse = np.array(rmse_list, dtype = float)
    save_rmse = pd.DataFrame(save_rmse)
    save_rmse.to_csv('./match_results/' + dataset + threhods + str(epoch) + '_rmse.csv', header = None)
    cv2.imwrite('./visual_result/' + dataset + threhods + '_' +str(epoch) + '_matched_poI1.bmp', combine_img)

    print('The Rmse is:',rmse_list)
    print('The Rmseloo is:',rmse_loo_list)
    return rmse, R_loo
