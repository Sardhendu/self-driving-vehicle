

# def check_if_parallel()
#
# import numpy as np
# import random
# from collections import deque
#
# def test_queue(max_size=10):
#     d = deque([], max_size)
#     for i in np.arange(1, 2*max_size):
#         print('Running i =====================>, ', i)
#         r = np.random.random((1,2))
#         d.append(r)
#         print("dumping :", r)
#         print(r[0])
#
#
#     # while d:
#     #     print("Popping ======> ")
#     #     print (d.popleft())
#
#
# test_queue()
# def check_if_parallel(left_lane_x_points, right_lane_x_points, y_points):
import numpy as np

720, 1280


a = np.tile(np.linspace(1, 7, 720).reshape(-1, 1), 1280)

# b = np.linspace(1, 2, 160)      # 0-200
# c = np.linspace(2, 3, 160)      # 200-400
# d = np.linspace(3, 2, 160)      # 400-600
# e = np.linspace(2, 1, 160)      # 600-700
# f = np.linspace(1, 2, 160)      # 700-800
# g = np.linspace(2, 3, 160)      # 800-1000
# h = np.linspace(3, 2, 160)      # 1000-1200
# i = np.linspace(2, 1, 160)

ll = np.tile(np.concatenate((
    np.linspace(1, 2, 160),
    np.linspace(2, 3, 160),
    np.linspace(3, 2, 160),
    np.linspace(2, 1, 160),
    np.linspace(1, 2, 160),
    np.linspace(2, 3, 160),
    np.linspace(3, 2, 160),
    np.linspace(2, 1, 160)
)).reshape(-1, 1), 720).T

hist_weight_matrix = ll*a
# print(ll.shape)
# print(a.shape)
# print(a)

print(hist_weight_matrix.shape)

print(np.sum(hist_weight_matrix))
hist_weight_matrix /= np.sum(hist_weight_matrix)



print(np.sum(hist_weight_matrix))

import matplotlib.pyplot as plt

plt.imshow(hist_weight_matrix)
plt.show()
