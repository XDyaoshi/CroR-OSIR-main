import t_demo as t
import test_made as tem
import trian_made as trm
import trian_made_2 as trm2
import SIFT_demo as sd
import R_SIFT_demo as rsd
import numpy as np
import os
import shutil


def delete_files_in_folder(folder_path):

    if not os.path.exists(folder_path):
        print(f"there is no {folder_path} ")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  
                # print(f"complete to delete: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  
                # print(f"complete to delete: {item_path}")
        except Exception as e:
            print(f"fail to delete {item_path}: {e}")


# This folder is used to generate the dataset for the second work
def made_data(data_name = "yellowa", image_last_name = "bmp",T0 = [[0.9973,0.0011,-9.9636], [-0.0011, 0.9973,-0.5292],[0,0,1]]):

    # The sensed image
    image_name = "sensed" + "." + image_last_name

    # The referenced image
    img_name = "reference" + "." + image_last_name

    dir_path ='data/' + data_name

    # Initial transformation matrix

    T0 = np.array(T0)
    # T0 = T0.T
    
    # Step 0: Delete old files
    folder_path_train = dir_path + '/train'
    folder_path_test_R = dir_path + '/test_R'
    folder_path_test_S = dir_path + '/test_S'
    delete_files_in_folder(folder_path_train)
    delete_files_in_folder(folder_path_test_R)
    delete_files_in_folder(folder_path_test_S)


    file_sen = dir_path + '/sen_data.csv'
    file_sen_n = dir_path + '/sen_n_data.csv'
    file_sen_r = dir_path + '/sen_r_data.csv'
    file_ref = dir_path + '/ref_data.csv'
    if os.path.exists(file_sen):
        os.remove(file_sen)

    if os.path.exists(file_sen_n):
        os.remove(file_sen_n)

    if os.path.exists(file_sen_r):
        os.remove(file_sen_r)

    if os.path.exists(file_ref):
        os.remove(file_ref)
    # Step 1: Find keypoints using SIFT algorithm on the sensed image

    sd.run_demo(data_name, image_name, dir_path, Max_keypoints = 2000, Distance = 0, flag = 0, edge_margin = 115)

    # Step 2:, the sought point is transformed into the corresponding
    # point on the reference image through the transformation matrix T0

    t.run_demo(data_name,img_name,T0,dir_path)

    # Step 3:, find points on the reference image

    rsd.run_demo(data_name, img_name, dir_path, Max_keypionts = 2000, Distance = 0, flag = 1)

    # Step 4: Make the test dataset

    tem.run_demo(data_name,img_name,image_last_name,dir_path = dir_path)
    tem.run_demo_S(data_name,img_name,image_last_name,dir_path = dir_path)

    # Step 5: Make the train dataset

    for count in range(0,5):
        print(count)
        trm.run_demo(data_name,img_name,image_last_name,dir_path = dir_path,knn = 1,count = count) 
        trm2.run_demo(data_name,img_name,image_last_name,dir_path = dir_path,knn = 1,count = count) 

