import pandas as pd
import numpy as np
import cv2


def run_demo(data_name,img_name,T0,dir_path):

    path = dir_path + "/"
    data = pd.read_csv(path + "sen_data.csv",header = None,dtype = int,sep = ',')
    data = np.array(data)
    data = data[1:]

    change_point = np.zeros([len(data),2])
    sen_path = path + img_name
    img = cv2.imread(sen_path)
    height, weight = img.shape[:2]

    delete_index = []
    point = np.ones([data.shape[0],1])
    point =  np.concatenate((data,point),axis = 1)

    for i in range(len(data)):
 
        temp = np.dot(point[i,:], T0.T)
    
        temp /= temp[2:3]
        temp = np.around(temp)
        
        x = temp[0]
        y = temp[1]
        nmsize = 16
        if x - nmsize > 0 and (x + nmsize) < weight and y - nmsize > 0 and (y + nmsize) < height:

              change_point[i, 0] = temp[0]
              change_point[i, 1] = temp[1]
        else:
            delete_index.append(i)

    data = np.delete(data, delete_index, axis=0)
    change_point = np.delete(change_point, delete_index, axis=0)

    path2 = dir_path + "/"
    save = pd.DataFrame(change_point)
    save.to_csv(path2 + 'sen_r_data.csv', index=False, header=False)
    save = pd.DataFrame(data)
    save.to_csv(path2 + 'sen_n_data.csv', index=False, header=False)
