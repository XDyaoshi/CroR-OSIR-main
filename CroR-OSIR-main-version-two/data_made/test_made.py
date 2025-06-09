import cv2
import pandas as pd
import numpy as np
import os

def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


def run_demo(data_name,img_name,image_last_name,dir_path):

    path = dir_path + "/"
    data = pd.read_csv(path+"ref_data.csv",header=None)
    data = np.array(data)
    train_save_path = dir_path + "/test_S/"
    ref_path = path + img_name

    img = cv2.imread(ref_path)
    enlarge = np.ones((img.shape[0]+130,img.shape[1]+130,3), np.uint8) * 255
    enlarge[65:(enlarge.shape[0]-65),65:(enlarge.shape[1]-65),:] = img
    
    for size in range (64,65,8):

        file_path = train_save_path
        create_dir_not_exist(file_path)
        nsize=int(size/2)

        for i in range(len(data)-1):

            x = int(data[i+1,0]) + 65 
            y = int(data[i+1,1]) + 65
            cropImg = enlarge[y-nsize:y+nsize, x-nsize:x+nsize]  
            cropImg = cv2.resize(cropImg,(64,64))
            if not os.path.exists(file_path):
               os.mkdir(file_path)
            create_dir_not_exist(file_path + "/" + str(i+1))
            cv2.imwrite(file_path + "/" + str(i+1) + "/"+str(i+1) + "." + image_last_name, cropImg)


def run_demo_S(data_name,img_name,image_last_name,dir_path):

    path = dir_path + "/"
    data = pd.read_csv(path+"sen_r_data.csv",header=None)
    data = np.array(data)
    data2 = pd.read_csv(path+"ref_data.csv",header=None)
    data2 = np.array(data2)
    size2 = len(data2) - 1

    train_save_path = dir_path + "/test_R/"
    ref_path = path + img_name
    img = cv2.imread(ref_path)
    enlarge = np.ones((img.shape[0]+130,img.shape[1]+130,3), np.uint8) * 255
    enlarge[65:(enlarge.shape[0]-65),65:(enlarge.shape[1]-65),:] = img
    
    for size in range (64,65,8):

        file_path = train_save_path
        create_dir_not_exist(file_path)
        nsize=int(size/2)

        for i in range(len(data)):

            x = int(data[i,0]) + 65 
            y = int(data[i,1]) + 65
            cropImg = enlarge[y-nsize:y+nsize, x-nsize:x+nsize]  
            cropImg = cv2.resize(cropImg,(64,64))
            if not os.path.exists(file_path):
               os.mkdir(file_path)
            create_dir_not_exist(file_path + "/" + str(i + 1 + size2))
            cv2.imwrite(file_path + "/" + str(i + 1 + size2) + "/"+ str(i + 1 + size2) + "." + image_last_name, cropImg)
