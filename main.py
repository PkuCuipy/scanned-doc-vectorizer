# 2023-03-31(19.59.05)
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import trange, tqdm
from collections import namedtuple

# 设置 plt 窗口大小
plt.rcParams["figure.figsize"] = (13, 9)

# 读入图片, 转为灰度图, 然后二值化
grey_img = np.array(cv2.imread("data/test_12.png", cv2.IMREAD_GRAYSCALE))
bin_threshold, bin_img = cv2.threshold(grey_img, 175, 255, cv2.THRESH_BINARY_INV)   # 前景为白色 (255)
plt.imshow(bin_img, cmap="gray")

# 提取所有的连通区域
nr_labels, label_map, label_stats, label_centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=4)    # 使用 4-连通, 尽可能缩小每个区域
plt.imshow(label_map, cmap="jet")

# 保存每个连通区域的信息
Area = namedtuple("Area", ["idx", "x", "y", "w", "h", "cx", "cy", "area", "bin_mask", "grey_map"])
areas = []
for idx in range(1, nr_labels):     # 0 为背景, 忽略
    x, y, w, h, area = label_stats[idx]
    centroid = label_centroids[idx] - [x, y]
    view = (slice(y, y + h), slice(x, x + w))           # x 向右, y 向下, 因此 y 才是 i, x 才是 j!
    bin_mask = (label_map[view] == idx)
    grey_map = np.where(bin_mask, grey_img[view], 255)  # fixme: 1.这里假设了背景色是 255;  2.被 bin_mask 约束使得遗漏浅色抗锯齿部分
    areas.append(Area(idx=idx, x=x, y=y, w=w, h=h, cx=centroid[0], cy=centroid[1], area=area, bin_mask=bin_mask, grey_map=grey_map))

# 绘制每个连通域
for area in tqdm(areas):
    plt.gca().add_patch(plt.Rectangle((area.x - 0.5, area.y - 0.5), area.w, area.h, fill=False, edgecolor="r", linewidth=0.5))    # 绘制 bbox
    plt.gca().add_patch(plt.Circle((area.x + area.cx, area.y + area.cy), radius=0.5, fill=False, edgecolor="g", linewidth=0.5))   # 绘制 centroid
    cv2.imwrite(f"result/test_12_area_{area.idx}.png", area.grey_map)   # 保存图片
plt.show()


# 亚像素图片移动
def move_img_subpixel(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    assert abs(dx) < 1.0 and abs(dy) < 1.0
    h, w = img.shape
    padded = np.pad(img, 1, mode="edge")
    x, y = np.meshgrid(np.arange(1, w + 1, dtype="f4") - dx, np.arange(1, h + 1, dtype="f4") - dy)
    return cv2.remap(padded, x, y, cv2.INTER_LINEAR)

def test_move_img_subpixel(img_path, save_folder):
    img3 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    for delta in np.linspace(0, 0.99, 10):
        cv2.imwrite(f"{save_folder}/test_13_moved_{delta}.png", move_img_subpixel(img3, delta, delta))


# 在高分辨率下, 将两个区域的 centroid 对齐
def align_centroids(a1: Area, a2: Area, upscale: int, canvas_w_prompt=None, canvas_h_prompt=None) -> np.ndarray:
    # Note: 目前的实现方式是 先双线性超采样, 然后再移动. 这样涉及两次近似. todo: 可以基于 cv2.remap 实现成仅需一次近似.
    assert upscale > 0 and upscale % 1 == 0
    canvas_w = int(max(a1.cx, a1.w - a1.cx, a2.cx, a2.w - a2.cx) + 2) * 2 * upscale     # +2: 留出余量空间
    canvas_h = int(max(a1.cy, a1.h - a1.cy, a2.cy, a2.h - a2.cy) + 2) * 2 * upscale
    if canvas_w_prompt > canvas_w: canvas_w = canvas_w_prompt
    if canvas_h_prompt > canvas_h: canvas_h = canvas_h_prompt
    # 在超采样前, 先在周围围一圈白色
    padded_a1 = np.pad(a1.grey_map, 1, mode='constant', constant_values=(255, 255))
    padded_a2 = np.pad(a2.grey_map, 1, mode='constant', constant_values=(255, 255))
    upscaled_padded_a1 = cv2.resize(padded_a1, (0, 0), fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)
    upscaled_padded_a2 = cv2.resize(padded_a2, (0, 0), fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)
    # 超采样之后的 centroid 坐标
    upscaled_padded_a1_cx = (a1.cx + 1) * upscale
    upscaled_padded_a1_cy = (a1.cy + 1) * upscale
    upscaled_padded_a2_cx = (a2.cx + 1) * upscale
    upscaled_padded_a2_cy = (a2.cy + 1) * upscale
    # 我们要将超采样后的 up_a1 和 up_a2 绘制在同一个 canvas 上, 使得它们的 centroid 对齐, 因此分别计算需要移动的量
    move_1_x = canvas_w / 2 - upscaled_padded_a1_cx
    move_1_y = canvas_h / 2 - upscaled_padded_a1_cy
    move_2_x = canvas_w / 2 - upscaled_padded_a2_cx
    move_2_y = canvas_h / 2 - upscaled_padded_a2_cy
    # 将移动量转换为整数和小数部分, 整数部分用于数组级别的移动, 小数部分用于亚像素级别的移动
    move_1_x_int = np.round(move_1_x).astype("i4");   move_1_x_f32 = move_1_x - move_1_x_int
    move_1_y_int = np.round(move_1_y).astype("i4");   move_1_y_f32 = move_1_y - move_1_y_int
    move_2_x_int = np.round(move_2_x).astype("i4");   move_2_x_f32 = move_2_x - move_2_x_int
    move_2_y_int = np.round(move_2_y).astype("i4");   move_2_y_f32 = move_2_y - move_2_y_int
    # 初始化两个 area 的 canvas
    canvas_a1 = np.ones((canvas_h, canvas_w), dtype="f4") * 255   # ones * 255: 初始化背景为白色
    canvas_a2 = np.ones((canvas_h, canvas_w), dtype="f4") * 255
    # 整数级别的移动
    canvas_a1[move_1_y_int : move_1_y_int + upscaled_padded_a1.shape[0], move_1_x_int : move_1_x_int + upscaled_padded_a1.shape[1]] = upscaled_padded_a1
    canvas_a2[move_2_y_int : move_2_y_int + upscaled_padded_a2.shape[0], move_2_x_int : move_2_x_int + upscaled_padded_a2.shape[1]] = upscaled_padded_a2
    # 亚像素级别的移动
    canvas_a1 = move_img_subpixel(canvas_a1, move_1_x_f32, move_1_y_f32)
    canvas_a2 = move_img_subpixel(canvas_a2, move_2_x_f32, move_2_y_f32)
    # 返回两个 area 的平均值
    return (canvas_a1 + canvas_a2) / 2


# 在高分辨率下, 将两个区域的 centroid 对齐
def move_img_to_target_canvas_with_centroid_aligned_and_up_scaled(a1: Area, upscale: int, canvas_w_prompt=None, canvas_h_prompt=None) -> np.ndarray:
    # Note: 目前的实现方式是 先双线性超采样, 然后再移动. 这样涉及两次近似. todo: 可以基于 cv2.remap 实现成仅需一次近似.
    canvas_w_at_least = int(max(a1.cx, a1.w - a1.cx) + 2) * 2 * upscale     # +2: 留出余量空间
    canvas_h_at_least = int(max(a1.cy, a1.h - a1.cy) + 2) * 2 * upscale
    assert upscale > 0 and upscale % 1 == 0
    assert canvas_w_prompt >= canvas_w_at_least
    assert canvas_h_prompt >= canvas_h_at_least
    # 在超采样前, 先在周围围一圈白色
    padded_a1 = np.pad(a1.grey_map, 1, mode='constant', constant_values=(255, 255))
    upscaled_padded_a1 = cv2.resize(padded_a1, (0, 0), fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)
    # 超采样之后的 centroid 坐标
    upscaled_padded_a1_cx = (a1.cx + 1) * upscale
    upscaled_padded_a1_cy = (a1.cy + 1) * upscale
    # 为了将超采样后的 up_a1 和 canvas 的 centroid 对齐, 计算需要移动的量
    move_1_x = canvas_w_prompt / 2 - upscaled_padded_a1_cx
    move_1_y = canvas_h_prompt / 2 - upscaled_padded_a1_cy
    # 将移动量转换为整数和小数部分, 整数部分用于数组级别的移动, 小数部分用于亚像素级别的移动
    move_1_x_int = np.round(move_1_x).astype("i4");   move_1_x_f32 = move_1_x - move_1_x_int
    move_1_y_int = np.round(move_1_y).astype("i4");   move_1_y_f32 = move_1_y - move_1_y_int
    # 初始化 area 的 canvas
    canvas_a1 = np.ones((canvas_h_prompt, canvas_w_prompt), dtype="f4") * 255   # ones * 255: 初始化背景为白色
    # 整数级别的移动
    canvas_a1[move_1_y_int : move_1_y_int + upscaled_padded_a1.shape[0], move_1_x_int : move_1_x_int + upscaled_padded_a1.shape[1]] = upscaled_padded_a1
    # 亚像素级别的移动
    canvas_a1 = move_img_subpixel(canvas_a1, move_1_x_f32, move_1_y_f32)
    return canvas_a1


# 测试 align_centroids()
for similar_indices in [
    [261,454,506,617,640,169,211,224,247,271,304,423,429,460,515,517,527,528,583,634,643,163,164,410,378,207,317,500,503,649,177],   # `e`
    [209,268,350,380,381,416,421,424,536,566,618,],    # `a`
    [242,243,490,495,497,611,612,616,349,447,347,494,491,446,397,346,244,191,559,555,398,291,292,348,396,492,557,613,614,],   # `t`
]:
    similar_areas = [areas[i - 1] for i in similar_indices]
    all_aligned = [move_img_to_target_canvas_with_centroid_aligned_and_up_scaled(a, upscale=10, canvas_h_prompt=300, canvas_w_prompt=300) for a in similar_areas]
    # 打印每个对齐后的图像
    # for mat in all_aligned:
    #     plt.imshow(mat, cmap="gray")
    #     plt.show()
    # 对 all_aligned 进行平均
    all_aligned_concat = np.array(all_aligned)
    all_aligned_concat = np.mean(all_aligned_concat, axis=0)
    plt.imshow(all_aligned_concat, cmap="gray")
    plt.show()
    # 二值化
    # plt.imshow(all_aligned_concat > 100, cmap="gray")










