import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''The program is responsible for updating the T-matrix during iteration'''
def print_transform(epoch, begin_epoch, matrix):
    
    print(f"┌{'─'*40}┐")
    print(f"│ Transform Matrix [Epoch {epoch}]")
    print(f"├{'─'*40}┤")
    print(np.array2string(matrix, 
                         formatter={'float_kind':lambda x: "%.3f" % x},
                         prefix="│ ",
                         max_line_width=100))
    print(f"└{'─'*40}┘")

def compute_homography_lstsq(dataset = 'yama', threhods = 0.95, epoch = 5, begin_epoch = 5):
    r_root = './data/' + dataset + '/ref_data.csv'
    s_root = './data/' + dataset + '/sen_n_data.csv'

    r_point = pd.read_csv(r_root, header = None, dtype = int)
    s_point = pd.read_csv(s_root, header = None, dtype = int)
    r_point = np.array(r_point)
    r_point = r_point[1:]
    s_point = np.array(s_point)
    s_point_unchange = np.array(s_point)

    rmse_list = []
    rmse_loo_list = []

    tensor_mix = torch.load('./tensor_result/' + dataset + '_' + str(threhods) + '_' + 'tensor_mix' + str(epoch) +'.pt')
    
    print('The number of point pairs found in this iteration is:', len(tensor_mix))

    ref_list = []
    sen_list = []
    sen_list_unchange = []

    for item in tensor_mix:
        ref_list.append(r_point[item[0].int() - 1].tolist())
        sen_list.append(s_point[item[1].int() - len(r_point) - 1].tolist())
        sen_list_unchange.append(s_point_unchange[item[1].int() - len(r_point) - 1].tolist())

    save_ref = np.array(ref_list)
    save_sen = np.array(sen_list_unchange)
    save_ref = pd.DataFrame(save_ref)
    save_sen = pd.DataFrame(save_sen)
    save_ref.to_csv('./match_results/' + dataset + str(threhods) + '_' + str(epoch) + '_matched_ref_points.csv', header = None)
    save_sen.to_csv('./match_results/' + dataset + str(threhods) + '_' + str(epoch) + '_matched_sen_points.csv', header = None)
    ref_array = np.array(ref_list, dtype = float)
    sen_array = np.array(sen_list, dtype = float)
    num_points = len(ref_array)
    if num_points < 4:
        raise ValueError("至少需要4对匹配点来求解单应性矩阵")

    A = []
    for i in range(num_points):
        x, y = sen_array[i]
        x_prime, y_prime = ref_array[i]

        A.append([-x, -y, -1,  0,  0,  0, x*x_prime, y*x_prime, x_prime])
        A.append([ 0,  0,  0, -x, -y, -1, x*y_prime, y*y_prime, y_prime])

    A = np.array(A, dtype=np.float32)

    # 计算 SVD
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)  # 取最小奇异值对应的特征向量

    return H / H[2, 2]  # 归一化，使 H[2,2] = 1



def cal_T_main(dataset = 'yama', threhods = 0.95, epoch = 5, begin_epoch = 5):
    r_root = './data/' + dataset + '/ref_data.csv'
    s_root = './data/' + dataset + '/sen_n_data.csv'

    r_point = pd.read_csv(r_root, header = None, dtype = int)
    s_point = pd.read_csv(s_root, header = None, dtype = int)
    r_point = np.array(r_point)
    r_point = r_point[1:]
    s_point = np.array(s_point)
    s_point_unchange = np.array(s_point)

    rmse_list = []
    rmse_loo_list = []

    tensor_mix = torch.load('./tensor_result/' + dataset + '_' + str(threhods) + '_' + 'tensor_mix' + str(epoch) +'.pt')
    
    print('The number of point pairs found in this iteration is:', len(tensor_mix))

    ref_list = []
    sen_list = []
    sen_list_unchange = []

    for item in tensor_mix:
        ref_list.append(r_point[item[0].int() - 1].tolist())
        sen_list.append(s_point[item[1].int() - len(r_point) - 1].tolist())
        sen_list_unchange.append(s_point_unchange[item[1].int() - len(r_point) - 1].tolist())

    save_ref = np.array(ref_list)
    save_sen = np.array(sen_list_unchange)
    save_ref = pd.DataFrame(save_ref)
    save_sen = pd.DataFrame(save_sen)
    save_ref.to_csv('./match_results/' + dataset + str(threhods) + '_' + str(epoch) + '_matched_ref_points.csv', header = None)
    save_sen.to_csv('./match_results/' + dataset + str(threhods) + '_' + str(epoch) + '_matched_sen_points.csv', header = None)
    ref_array = np.array(ref_list, dtype = float)
    sen_array = np.array(sen_list, dtype = float)
    tool_martix = np.ones((len(sen_list),1),dtype=float)
    sen_array = np.concatenate((sen_array, tool_martix), axis=1, dtype = float)
    ref_array = np.concatenate((ref_array, tool_martix), axis=1, dtype = float)
    matrix_1 = np.dot(sen_array.T,sen_array)
    matrix_1 = np.linalg.inv(matrix_1)
    matrix_2 = np.dot(sen_array.T,ref_array)
    matrix_trans = np.dot(matrix_1,matrix_2)
    print_transform(epoch = epoch, begin_epoch = begin_epoch, matrix = matrix_trans.T)
    return matrix_trans.T
