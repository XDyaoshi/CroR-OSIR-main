import oldemo.t_demo as t
import oldemo.test_made as tem
import oldemo.trian_made as trm
import oldemo.trian_made_2 as trm2
import numpy as np
import oldemo.SIFT_demo as sd
import oldemo.R_SIFT_demo as rsd

# This folder is used to generate the dataset for the second work

data_name = "yellowA"

# The sensed image

image_name = "I2.bmp"

# The referenced image

img_name = "I1.bmp"
image_last_name = "bmp"
dir_path ='data/' + data_name

# Initial transformation matrix

T0 = [[1.0003,-0.0015,8.9882], [0.0015,1.0003,49.0823], [0,0,1]] #yellowR1(Derived from SIFT)
T0 = np.array(T0)
# T0 = T0.T

# Step 1: Find keypoints using SIFT algorithm on the sensed image

sd.run_demo(data_name, image_name, dir_path, Max_keypionts = 2000, Distance = 0, flag = 0)

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