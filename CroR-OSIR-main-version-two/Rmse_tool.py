import numpy as np
import pandas as pd
import math
import os
import time
import scipy.stats as stats
'''This code is for calculating evaluation metrics.'''

def rmse(refpoint,senpoint,matrix):
    senpoint_t = np.dot(senpoint,matrix)
        
    for item in senpoint_t:
        item[0] = item[0] / item[2]
        item[1] = item[1] / item[2]

    cal_martix = refpoint-senpoint_t
    cal_martix = cal_martix[:,:2]
    # print(cal_martix)
    res = 0
    for i in range(len(cal_martix)):
        res = res + math.sqrt(cal_martix[i,0]**2 + cal_martix[i,1]**2)
    res = res/len(cal_martix)
    return res

