import copy
import math

import numpy as np
import random


# a = np.array([[1,2],
#               [3,4],
#               [6,5],
#               [12,13],
#               [11,19]])
# maxMakespan = np.max(a, axis=0)  # 输出 6
# maxTEC = np.max(a, axis=1)  # 输出 6
# centroid1 = np.mean(a, axis=0)  # 按列求均值
# a = 1

# arr1 = np.array([[1,2,3],
#                  [4,5,6],
#                  [7,8,9]])
# arr2 = np.array([[1,2,3],
#                  [4,5,6],
#                  [7,8,9]])
# combined_array = np.concatenate([arr1[:, 2], arr2[:, 2]])

# arr1 = np.array([10, 9, 1])
# arr2 = np.array([8, 9, 1])
# a = (arr1[:2] == arr2[:2])
# a = np.any(arr1[:2] == arr2[:2])
# # if np.any(arr1[:2] == arr2[:2]):
#
# a = 1
# counts = 0
# ps = 60
# arr = np.zeros(6)
# print(arr)
# for i in range(len(arr)):
#     # print(i)
#     if i < len(arr):
#         K = math.ceil(ps / len(arr))
#         print(f'K == {K}')
#     else:
#         # 最后一组生成剩余个体
#         K = ps - counts
#         print(f'K == {K}')
#     counts = counts + K
