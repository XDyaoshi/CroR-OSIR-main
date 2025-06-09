import cv2
import numpy as np
import pandas as pd
# def run_demo(data_name,image_name,dir_path,Max_keypionts = 500,Distance = 20, flag = 0):
 
#     image = cv2.imread(dir_path + "/" + image_name, cv2.COLOR_BGR2GRAY)  

#     sift = cv2.SIFT_create()


#     max_keypoints = Max_keypionts

#     keypoints = sift.detect(image, None)

#     keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_keypoints]

#     filtered_keypoints = []
#     for kp1 in keypoints:
#         is_valid = True
#         for kp2 in filtered_keypoints:
 
#             distance = int(cv2.norm(np.array(kp1.pt) - np.array(kp2.pt)))
#             if distance < Distance:
#                is_valid = False
#                break
#         if is_valid:
#             filtered_keypoints.append(kp1)
#     image_with_keypoints = cv2.drawKeypoints(image, filtered_keypoints, None)
#     sen_data = []
#     for kp in filtered_keypoints:
#         kp_pt = np.array(kp.pt)
#         kp_pt = np.around(kp_pt)
#         sen_data.append(kp_pt)
#     save = pd.DataFrame(sen_data, columns = [0, 1],dtype=int)
#     save = save.drop_duplicates()
#     if (flag == 0):
#         save.to_csv(dir_path +'/sen_data.csv',index=False,header=True)
#     else:
#         save.to_csv(dir_path +'/ref_data.csv',index=False,header=True)


# def run_demo(data_name, image_name, dir_path, 
#              Max_keypionts=500, Distance=20, flag=0, 
#              edge_margin = 55): 
    
#     image = cv2.imread(dir_path + "/" + image_name, cv2.IMREAD_GRAYSCALE)
    
#     height, width = image.shape
#     mask = np.zeros((height, width), dtype=np.uint8)
    
#     y_start = edge_margin
#     y_end = height - edge_margin
#     x_start = edge_margin
#     x_end = width - edge_margin
#     mask[y_start:y_end, x_start:x_end] = 255

#     sift = cv2.SIFT_create()
#     keypoints = sift.detect(image, mask)  
    
#     keypoints = sorted(keypoints, key=lambda x: -x.response)[:Max_keypionts]
    
#     filtered_keypoints = []
#     for kp1 in keypoints:
#         is_valid = True
#         for kp2 in filtered_keypoints:
#             distance = int(cv2.norm(np.array(kp1.pt) - np.array(kp2.pt)))
#             if distance < Distance:
#                 is_valid = False
#                 break
#         if is_valid:
#             filtered_keypoints.append(kp1)
 
#     sen_data = [np.around(np.array(kp.pt)).astype(int) for kp in filtered_keypoints]
#     save = pd.DataFrame(sen_data, columns=[0, 1]).drop_duplicates()
    
#     output_name = '/sen_data.csv' if flag == 0 else '/ref_data.csv'
#     save.to_csv(dir_path + output_name, index=False, header=True)


def run_demo(data_name, image_name, dir_path, Max_keypoints=500, Distance=20, flag=0, edge_margin=55): 

    image = cv2.imread(dir_path + "/" + image_name, cv2.IMREAD_GRAYSCALE)
    
    height, width = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)  # 先检测所有的关键点

    # 根据响应强度排序并选择最大数量的关键点
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:Max_keypoints]
    # 过滤掉位于边缘区域外的关键点
    filtered_keypoints = []
    for kp in keypoints:
        x, y = kp.pt
        if edge_margin <= x < width - edge_margin and edge_margin <= y < height - edge_margin:
            filtered_keypoints.append(kp)
    
    # 过滤掉距离过近的关键点
    final_keypoints = []
    for kp1 in filtered_keypoints:
        is_valid = True
        for kp2 in final_keypoints:
            distance = int(cv2.norm(np.array(kp1.pt) - np.array(kp2.pt)))
            if distance < Distance:
                is_valid = False
                break
        if is_valid:
            final_keypoints.append(kp1)

    # 保存关键点数据
    sen_data = [np.around(np.array(kp.pt)).astype(int) for kp in final_keypoints]
    save = pd.DataFrame(sen_data, columns=[0, 1]).drop_duplicates()
    
    output_name = '/sen_data.csv' if flag == 0 else '/ref_data.csv'
    save.to_csv(dir_path + output_name, index=False, header=True)