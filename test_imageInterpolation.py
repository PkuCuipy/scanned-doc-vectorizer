# 2023-04-01(00.17.46)
# 基于向量化的 Numpy 实现 Bilinear 插值, 并和 OpenCV 的结果进行比较

import numpy as np
import cv2
from tqdm import trange, tqdm


# 读入图片, 使用 OpenCV 的 Bilinear 插值到 100 倍大
img = cv2.imread("data/test_2x2.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (0, 0), fx=100, fy=100, interpolation=cv2.INTER_LINEAR)

# save image
cv2.imwrite("data/test_2x2_100x100_OpenCV.png", img)


# 将图像上所有像素的x坐标都减去0.5，向右移动0.5个像素
img2 = cv2.imread('./data/test_4x4.png')
h, w = img2.shape[:2]
# x, y = np.meshgrid(np.linspace(-0.5, w-1+0.5, 100), np.linspace(-0.5, h-1+0.5, 100))
x, y = np.meshgrid(np.linspace(0.5, 2.5, 100), np.linspace(0.5, 2.5, 100))
x = x.astype(np.float32)
y = y.astype(np.float32)

# 使用双线性插值对新位置不在整数位置上的像素进行插值
# img2 = cv2.remap(img2, x - 0.5, y, cv2.INTER_LINEAR)
# img2 = cv2.remap(img2, x + 0.5, y, cv2.INTER_LINEAR)
img2 = cv2.remap(img2, x, y, cv2.INTER_LINEAR)

# 显示结果
# cv2.imwrite("data/test_13_moved.png", img2)





