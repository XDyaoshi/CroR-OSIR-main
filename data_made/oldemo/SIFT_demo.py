import cv2
import numpy as np
import pandas as pd
def run_demo(data_name,image_name,dir_path,Max_keypionts = 500,Distance = 20, flag = 0):
 
    image = cv2.imread(dir_path + "/" + image_name, cv2.COLOR_BGR2GRAY)  

    sift = cv2.SIFT_create()


    max_keypoints = Max_keypionts

    keypoints = sift.detect(image, None)

    keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_keypoints]

    filtered_keypoints = []
    for kp1 in keypoints:
        is_valid = True
        for kp2 in filtered_keypoints:
 
            distance = int(cv2.norm(np.array(kp1.pt) - np.array(kp2.pt)))
            if distance < Distance:
               is_valid = False
               break
        if is_valid:
            filtered_keypoints.append(kp1)
    image_with_keypoints = cv2.drawKeypoints(image, filtered_keypoints, None)
    sen_data = []
    for kp in filtered_keypoints:
        kp_pt = np.array(kp.pt)
        kp_pt = np.around(kp_pt)
        sen_data.append(kp_pt)
    save = pd.DataFrame(sen_data, columns = [0, 1],dtype=int)
    save = save.drop_duplicates()
    if (flag == 0):
        save.to_csv(dir_path +'/sen_data.csv',index=False,header=True)
    else:
        save.to_csv(dir_path +'/ref_data.csv',index=False,header=True)

    # print("The number of keypoint is:", len(filtered_keypoints))

