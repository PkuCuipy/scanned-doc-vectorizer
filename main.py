# 2023-03-31(19.59.05)
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import trange, tqdm
from collections import namedtuple
from typing import Iterable

# 存储区域信息
Area = namedtuple("Area", ["idx", "x", "y", "w", "h", "cx", "cy", "area", "bin_mask", "grey_map", "symbol_id"])

# 计算质心
def calc_centroid(mat: np.ndarray) -> np.ndarray:
    cols_sum = np.sum(mat, axis=1)
    rows_sum = np.sum(mat, axis=0)
    c_i = (cols_sum / np.sum(cols_sum)) @ np.arange(mat.shape[0])
    c_j = (rows_sum / np.sum(rows_sum)) @ np.arange(mat.shape[1])
    return np.array([c_j, c_i])

# 亚像素图片移动
def move_img_subpixel(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    assert abs(dx) < 1.0 and abs(dy) < 1.0
    h, w = img.shape
    padded = np.pad(img, 1, mode="edge")
    x, y = np.meshgrid(np.arange(1, w + 1, dtype="f4") - dx, np.arange(1, h + 1, dtype="f4") - dy)
    return cv2.remap(padded, x, y, cv2.INTER_NEAREST)

# 测试亚像素图片移动
def test_move_img_subpixel(img_path, save_folder):
    img3 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    for delta in np.linspace(0, 0.99, 10):
        cv2.imwrite(f"{save_folder}/test_13_moved_{delta}.png", move_img_subpixel(img3, delta, delta))

# 在高分辨率下, 将两个区域的 centroid 对齐
def move_img_to_target_canvas_with_centroid_aligned_and_up_scaled(a: Area, upscale: int, canvas_w: int = None, canvas_h : int = None) -> np.ndarray:
    # Note: 目前的实现方式是 先双线性超采样, 然后再移动. 这样涉及两次近似. todo: 可以基于 cv2.remap 实现成仅需一次近似.
    canvas_w_at_least = int(max(a.cx, a.w - a.cx) + 3) * 2 * upscale     # +3: 留出余量空间
    canvas_h_at_least = int(max(a.cy, a.h - a.cy) + 3) * 2 * upscale
    assert upscale > 0 and upscale % 1 == 0, "仅支持整数倍放大!"
    assert canvas_w >= canvas_w_at_least and canvas_h >= canvas_h_at_least, "提供的画布尺寸太小!"
    # 在超采样前, 先在周围围一圈白色
    padded_a = np.pad(a.grey_map, 1, mode='constant', constant_values=(255, 255))
    upscaled_padded_a = cv2.resize(padded_a, (0, 0), fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)   # fixme
    # 超采样之后的 centroid 坐标
    upscaled_padded_a_cx = ((a.cx + 1) * upscale + 0.5 * (upscale - 1))     # +1 是因为 padding, * s + 0.5(s-1) 是因为 upscale.
    upscaled_padded_a_cy = ((a.cy + 1) * upscale + 0.5 * (upscale - 1))     # 基于 NEAREST 可以轻易推导 (认为像素坐标点是撑满图片的小方块的中心点). 但对 LINEAR 好像也成立.
    # assert np.allclose([upscaled_padded_a_cx, upscaled_padded_a_cy], calc_centroid(255-upscaled_padded_a))   # 对 LINEAR 好像也成立的例证. 但对 CUBIC 就不成立了.
    # 为了将超采样后的 up_a1 和 canvas 的 centroid 对齐, 计算需要移动的量
    canvas_cx = (canvas_w - 1) / 2
    canvas_cy = (canvas_h - 1) / 2
    move_x = canvas_cx - upscaled_padded_a_cx
    move_y = canvas_cy - upscaled_padded_a_cy
    # 将移动量转换为整数和小数部分, 整数部分用于数组级别的移动, 小数部分用于亚像素级别的移动
    move_x_int = np.round(move_x).astype("i4")
    move_y_int = np.round(move_y).astype("i4")
    move_x_f32 = move_x - move_x_int
    move_y_f32 = move_y - move_y_int
    # 初始化 area 的 canvas
    can = np.ones((canvas_h, canvas_w), dtype="f4") * 255   # ones * 255: 初始化背景为白色
    # 整数级别的移动
    can[move_y_int : move_y_int + upscaled_padded_a.shape[0], move_x_int : move_x_int + upscaled_padded_a.shape[1]] = upscaled_padded_a
    # 亚像素级别的移动
    can = move_img_subpixel(can, move_x_f32, move_y_f32)
    return can

# 在高分辨率下, 将若干个区域的 centroid 对齐
def centroids_aligned_mean(areas: Iterable[Area], upscale: int) -> np.ndarray:
    # Note: 目前的实现方式是 先双线性超采样, 然后再移动. 这样涉及两次近似. todo: 可以基于 cv2.remap 实现成仅需一次近似.
    assert upscale > 0 and upscale % 1 == 0
    canvas_h = int(max(max(a.cy, a.h - a.cy) for a in areas) + 3) * 2 * upscale     # +3: 留出余量空间
    canvas_w = int(max(max(a.cx, a.w - a.cx) for a in areas) + 3) * 2 * upscale
    all_canvas = np.array([move_img_to_target_canvas_with_centroid_aligned_and_up_scaled(a, upscale, canvas_w, canvas_h) for a in areas])
    return np.mean(all_canvas, axis=0)      # min(), max(), median() 实验效果不佳

# 测试 centroids_aligned_mean()
def test_centroids_aligned_mean():
    for test_id, similar_indices in enumerate([
        # [54, 337, 1860, 735, 1403, 61, 1549, 305, 1137, 1322, 1604, 1739, 1740, 91, 306, 1406, 1556, 324, 761, 670, 667,675, 962, 1256, 1617, 126, 1816, 484, 473, 1261, 1280, 813, 835, 138, 1678, 1263, 83, 743, 690, 1339, 1027, 49,1383, 1407, 1422, 210, 1241, 1145, 1578, 1759, 810, 832, 1050, 838, 1272, 1445, 1610, 1544, 1068,], # `m`, test_9_down_from_1200
        # [2314, 2076, 2009, 2323, ], # `都`, test_6
        # [344, 976, 200, 878, 515, ], # `格`, test_15
        # [75, 74, 546, 435, 511, 489, 481, 140, 47, ], # `在`, test_15
        # [1528, 201, 996, 1374, 803, 1342, 909, 914, 573, 854, 149, 1408, 725, 553, 437, 224, 828, 1466, 282, 418, 1190, 44, 484, 842, 153, 580, 773, 1297, 1211, 590, 775, 360, 340, 746, 1500, 4, ], # `m`, test_14
    ]):
        upscale = 2                                                         # 超采样倍率
        similar_areas = [areas[i - 1] for i in similar_indices]
        grey_result = centroids_aligned_mean(similar_areas, upscale=upscale)
        plt.imshow(grey_result); plt.show()
        cv2.imwrite(f"test_centroids_aligned_mean_{test_id}_{len(similar_indices)}_up={upscale}.png", grey_result)

# ======================================================================================================================

if __name__ == "__main__":

    # 创建存放结果的文件夹
    if not os.path.exists("result"):
        os.mkdir("result")

    # 设置 plt 窗口大小
    plt.rcParams["figure.figsize"] = (13, 9)

    # 读入图片, 转为灰度图, 然后二值化
    grey_img = np.array(cv2.imread("data/test_14.png", cv2.IMREAD_GRAYSCALE))
    grey_img = cv2.resize(grey_img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR) # 在这里放大, 而不是在 align_centroids() 中, 因为这有利于提取区域(?)
    bin_threshold, bin_img = cv2.threshold(grey_img, 175, 255, cv2.THRESH_BINARY_INV)   # 前景为白色 (255)
    plt.imshow(bin_img, cmap="gray")
    plt.show()

    # 提取所有的连通区域
    nr_labels, label_map, label_stats, label_centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)    # fixme: 使用 4-连通, 尽可能缩小每个区域, 似乎更适合英文; 8-连通似乎适合中文
    plt.imshow(label_map, cmap="jet")

    # 保存每个连通区域的信息
    areas = []
    for idx in range(1, nr_labels):     # 0 为背景, 忽略
        x, y, w, h, area = label_stats[idx]
        view = (slice(y, y + h), slice(x, x + w))           # x 向右, y 向下, 因此 y 才是 i, x 才是 j!
        bin_mask = (label_map[view] == idx)
        grey_map = np.where(bin_mask, grey_img[view], 255)  # fixme: 1.这里假设了背景色是 255;  2.被 bin_mask 约束使得遗漏浅色抗锯齿部分
        centroid = calc_centroid(255 - grey_map)            # 基于灰度图计算, 此前是基于 bin_mask 计算的 (label_centroids[idx] - [x, y])
        areas.append(Area(idx=idx, x=x, y=y, w=w, h=h, cx=centroid[0], cy=centroid[1], area=area, bin_mask=bin_mask, grey_map=grey_map, symbol_id=None))

    # 绘制每个连通域
    for area in tqdm(areas):
        plt.gca().add_patch(plt.Rectangle((area.x - 0.5, area.y - 0.5), area.w, area.h, fill=False, edgecolor="r", linewidth=0.5))    # 绘制 bbox
        # plt.gca().add_patch(plt.Circle((area.x + area.cx, area.y + area.cy), radius=0.5, fill=False, edgecolor="g", linewidth=0.5))   # 绘制 centroid
        cv2.imwrite(f"result/{area.idx}.png", area.grey_map)   # 保存图片
    plt.show()

    # test_centroids_aligned_mean()

    # 2023-04-03 TODO:
    # 1. 创建 symbol_table, 每个 area 的 symbol_id 指向某个符号, 这个符号由 (grey_map, cx, cy) 表征, 其中 cx, cy 应该都是 int + 0.5 的样子 (因为 2x 超分后 canvas 是偶数边长的, 然后 centroid() 返回的 area 是重心中心化的).
    # 2. 构建一个 4x 大的空白图片, 把每个符号都放到这个图片上.
    # 3. 把每个符号的矢量勾勒出来, 构建一个 svg, 把每个符号放到 svg 上, 导出 svg.




