import os
import shutil
import cv2
import pandas as pd
import numpy as np

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


def run_demo(data_name,img_name,image_last_name,dir_path,knn,count):


    path = dir_path + "/"
    data = pd.read_csv(path+"sen_r_data.csv",header=None)
    data = np.array(data)
    data2 = pd.read_csv(path+"ref_data.csv",header=None)
    data2 = np.array(data2)
    patch_len = len(data2)
    train_save_path = dir_path + "/train/"
    ref_path = dir_path + "/" + img_name
    img = cv2.imread(ref_path)
    enlarge = np.ones((img.shape[0]+130,img.shape[1]+130,3), np.uint8) * 255
    enlarge[65:(enlarge.shape[0]-65),65:(enlarge.shape[1]-65),:] = img
    size = 66 - count
    knn = knn

    file_path = train_save_path
    create_dir_not_exist(file_path)
    nsize = int(size/2)

    for i in range(len(data)):

        create_dir_not_exist(file_path + str(i + patch_len))
        x=int(data[i,0])+65
        y=int(data[i,1])+65
        cropImg = enlarge[y-nsize:y+nsize, x-nsize:x+nsize]
        cropImg = cv2.resize(cropImg,(66,66))  
        cv2.imwrite(file_path+str(i + patch_len) +"/"+ str(count*9 + 1) + '.' +image_last_name, cropImg)
        x=int(data[i,0])+65 + knn
        y=int(data[i,1])+65
        cropImg = enlarge[y-nsize:y+nsize, x-nsize:x+nsize]

        cropImg = cv2.resize(cropImg,(66,66)) 
        cv2.imwrite(file_path+str(i + patch_len) +"/"+ str(count*9 + 2) + '.' + image_last_name, cropImg)
        x=int(data[i,0])+65 + knn
        y=int(data[i,1])+65 + knn
        cropImg = enlarge[y-nsize:y+nsize, x-nsize:x+nsize]  

        cropImg = cv2.resize(cropImg,(66,66))
        cv2.imwrite(file_path+str(i + patch_len) +"/"+ str(count*9 + 3) + '.' + image_last_name, cropImg)
        x=int(data[i,0])+65
        y=int(data[i,1])+65 + knn
        cropImg = enlarge[y-nsize:y+nsize, x-nsize:x+nsize]  

        cropImg = cv2.resize(cropImg,(66,66))
        cv2.imwrite(file_path+str(i + patch_len) +"/"+ str(count*9 + 4) + '.' + image_last_name, cropImg)
        x=int(data[i,0])+65 - knn
        y=int(data[i,1])+65 - knn
        cropImg = enlarge[y-nsize:y+nsize, x-nsize:x+nsize]  

        cropImg = cv2.resize(cropImg,(66,66))
        cv2.imwrite(file_path+str(i + patch_len) +"/"+ str(count*9 + 5) + '.' + image_last_name, cropImg)
        x=int(data[i,0])+65 - knn
        y=int(data[i,1])+65
        cropImg = enlarge[y-nsize:y+nsize, x-nsize:x+nsize]  

        cropImg = cv2.resize(cropImg,(66,66))
        cv2.imwrite(file_path+str(i + patch_len) +"/"+ str(count*9 + 6) + '.' + image_last_name, cropImg)
        x=int(data[i,0])+65
        y=int(data[i,1])+65 - knn
        cropImg = enlarge[y-nsize:y+nsize, x-nsize:x+nsize]  

        cropImg = cv2.resize(cropImg,(66,66)) 
        cv2.imwrite(file_path+str(i + patch_len) +"/"+ str(count*9 + 7) + '.' +image_last_name, cropImg)
        x=int(data[i,0])+65 + knn
        y=int(data[i,1])+65 - knn
        cropImg = enlarge[y-nsize:y+nsize, x-nsize:x+nsize]  

        cropImg = cv2.resize(cropImg,(66,66))
        cv2.imwrite(file_path+str(i + patch_len) +"/"+ str(count*9 + 8) + '.' +image_last_name, cropImg)
        x=int(data[i,0])+65 - knn
        y=int(data[i,1])+65 + knn
        cropImg = enlarge[y-nsize:y+nsize, x-nsize:x+nsize] 
        cropImg = cv2.resize(cropImg,(66,66))
        cv2.imwrite(file_path+str(i + patch_len) +"/"+ str(count*9 + 9) + '.' +image_last_name, cropImg)
