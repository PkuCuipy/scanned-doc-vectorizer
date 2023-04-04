# 2023-03-31(19.59.05)
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import trange, tqdm
from typing import Iterable

# 区域类
class Area:
    __slots__ = ["idx", "x", "y", "w", "h", "cx", "cy", "mass", "bin_mask", "grey_mask", "symbol_id", "feat_mat", "feat_mass"]
    def __init__(s, idx, x, y, w, h, cx, cy, mass, bin_mask, grey_mask):
        (s.idx, s.x, s.y, s.w, s.h, s.cx, s.cy, s.mass, s.bin_mask, s.grey_mask) = (idx, x, y, w, h, cx, cy, mass, bin_mask, grey_mask)

# 符号类
class Symbol:
    __slots__ = ["grey_mask", "cx", "cy"]
    def __init__(s, grey_mask, cx, cy):
        (s.grey_mask, s.cx, s.cy) = (grey_mask, cx, cy)

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
    return cv2.remap(padded, x, y, cv2.INTER_LINEAR)

# 测试亚像素图片移动
def test_move_img_subpixel(img_path, save_folder):
    img3 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    for delta in np.linspace(0, 0.99, 10):
        cv2.imwrite(f"{save_folder}/test_13_moved_{delta}.png", move_img_subpixel(img3, delta, delta))

# 一个质点(比如重心)在 upscale 后的新坐标
def upscaled_coord(x: float, upscale: int) -> float:
    return x * upscale + 0.5 * (upscale - 1)    # * s + 0.5(s-1) 基于 NEAREST 可以轻易推导 (认为像素坐标点是撑满图片的小方块的中心点). 但对 LINEAR 好像也成立.

# 一个 w, h 实心矩形的中心点坐标
def center_of_rect(s: int) -> float:
    return (s - 1) / 2

# 将一个 float 拆成 int + float
def float_split(x: float) -> (int, float):
    return (i := np.round(x).astype("i4")), x - i

# 将 area 超采样后, 将其重心移动到 target_canvas 的中心, 并返回 target_canvas
def move_img_to_target_canvas_with_centroid_aligned_and_up_scaled(a: Area, upscale: int, canvas_w: int, canvas_h : int) -> np.ndarray:
    # Note: 目前的实现方式是 先双线性超采样, 然后再移动. 这样涉及两次近似. todo: 可以基于 cv2.remap 实现成仅需一次近似.
    canvas_w_at_least = int(max(a.cx, a.w - a.cx) + 3) * 2 * upscale     # +3: 留出余量空间
    canvas_h_at_least = int(max(a.cy, a.h - a.cy) + 3) * 2 * upscale
    assert upscale > 0 and upscale % 1 == 0, "仅支持整数倍放大!"
    assert canvas_w >= canvas_w_at_least and canvas_h >= canvas_h_at_least, "提供的画布尺寸太小!"
    # 在超采样前, 先在周围围一圈黑色
    padded_a = np.pad(a.grey_mask, 1, mode='constant', constant_values=(0, 0))
    upscaled_padded_a = cv2.resize(padded_a, (0, 0), fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)   # fixme
    # 超采样之后的 centroid 坐标
    upscaled_padded_a_cx = upscaled_coord(a.cx + 1, upscale)  # +1 是因为 padding
    upscaled_padded_a_cy = upscaled_coord(a.cy + 1, upscale)
    assert np.allclose([upscaled_padded_a_cx, upscaled_padded_a_cy], calc_centroid(upscaled_padded_a))   # 对 LINEAR 好像也成立的例证. 但对 CUBIC 就不成立了.
    # 为了将超采样后的 up_a1 和 canvas 的 centroid 对齐, 计算需要移动的量
    canvas_cx = center_of_rect(canvas_w)
    canvas_cy = center_of_rect(canvas_h)
    move_x = canvas_cx - upscaled_padded_a_cx
    move_y = canvas_cy - upscaled_padded_a_cy
    # 将移动量转换为整数和小数部分, 整数部分用于数组级别的移动, 小数部分用于亚像素级别的移动
    move_x_int, move_x_f32 = float_split(move_x)
    move_y_int, move_y_f32 = float_split(move_y)
    # 初始化 area 的 canvas
    can = np.zeros((canvas_h, canvas_w), dtype="f4")   # 初始化背景为黑色
    # 整数级别的移动
    can[move_y_int : move_y_int + upscaled_padded_a.shape[0], move_x_int : move_x_int + upscaled_padded_a.shape[1]] = upscaled_padded_a
    # 亚像素级别的移动
    can = move_img_subpixel(can, move_x_f32, move_y_f32)
    return can

# 在高分辨率下, 将若干个区域的 centroid 对齐
def centroids_aligned_mean(areas: Iterable[Area], upscale: int, *, canvas_h_at_least_debug_prompt: int = 0, canvas_w_at_least_debug_prompt: int = 0) -> (np.ndarray, float, float):
    # Note: 目前的实现方式是 先双线性超采样, 然后再移动. 这样涉及两次近似. todo: 可以基于 cv2.remap 实现成仅需一次近似.
    # 以下会自行决定合适的 canvas 尺寸以保证能容纳所有重心对齐的 areas,
    # 但用户可以通过 canvas_h_at_least_debug_prompt 和 canvas_w_at_least_debug_prompt 来指定最小尺寸. (Debug 用)
    assert upscale > 0 and upscale % 1 == 0
    canvas_h = int(max(max(a.cy, a.h - a.cy) for a in areas) + 3) * 2 * upscale     # +3: 留出余量空间
    canvas_w = int(max(max(a.cx, a.w - a.cx) for a in areas) + 3) * 2 * upscale
    if canvas_h < canvas_h_at_least_debug_prompt: canvas_h = canvas_h_at_least_debug_prompt  # Debug
    if canvas_w < canvas_w_at_least_debug_prompt: canvas_w = canvas_w_at_least_debug_prompt  # Debug
    all_canvas = np.array([move_img_to_target_canvas_with_centroid_aligned_and_up_scaled(a, upscale, canvas_w, canvas_h) for a in areas])
    return np.mean(all_canvas, axis=0), (canvas_w-1) / 2, (canvas_h-1) / 2

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
        grey_result, _, _ = centroids_aligned_mean(similar_areas, upscale=upscale)
        plt.imshow(grey_result); plt.show()
        cv2.imwrite(f"test_centroids_aligned_mean_{test_id}_{len(similar_indices)}_up={upscale}.png", grey_result)

# 传入一个 area, 计算 feature_vector
def calc_feat_mat(area: Area) -> np.ndarray:
    feat_mat = centroids_aligned_mean([area, ], upscale=1, canvas_h_at_least_debug_prompt=80, canvas_w_at_least_debug_prompt=80)[0]
    feat_mat = cv2.resize(feat_mat, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    feat_mat = np.sqrt(feat_mat) / np.sqrt(255) * 255  # 实测这里加一个上凸函数 (开根乘十) 效果会好很多
    feat_mat = cv2.GaussianBlur(feat_mat, (3, 3), 0)
    return feat_mat

# 传入两个 area, 计算他们的 feature 差异度
def calc_feat_diff(a1: Area, a2: Area) -> float:
    # 目前差异度 (difference) 定义为: sum(|A-B|) / (sum(A)+sum(B)), 有种 IoU 的感觉, 只不过这里是 diff-over-union
    return np.sum(np.abs(a1.feat_mat - a2.feat_mat)) / ((a1.feat_mass + a2.feat_mass) / 2)

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
        grey_mask = np.where(bin_mask, 255 - grey_img[view], 0)  # 这里假设背景色是 0, 前景是 1~255;  fixme: 被 bin_mask 约束使得遗漏浅色抗锯齿部分
        centroid = calc_centroid(grey_mask)                      # 基于灰度图计算, 此前是基于 bin_mask 计算的 (label_centroids[idx] - [x, y])
        mass = np.sum(grey_mask)                                 # 基于灰度图计算, 此前是基于 bin_mask 计算的
        areas.append(Area(idx=idx, x=x, y=y, w=w, h=h, cx=centroid[0], cy=centroid[1], mass=mass, bin_mask=bin_mask, grey_mask=grey_mask))

    # 绘制每个连通域
    for area in tqdm(areas):
        plt.gca().add_patch(plt.Rectangle((area.x - 0.5, area.y - 0.5), area.w, area.h, fill=False, edgecolor="r", linewidth=0.5))    # 绘制 bbox
        # plt.gca().add_patch(plt.Circle((area.x + area.cx, area.y + area.cy), radius=0.5, fill=False, edgecolor="g", linewidth=0.5))   # 绘制 centroid
        # cv2.imwrite(f"result/{area.idx}.png", area.grey_mask)   # 保存图片
    plt.show()

    # 为每个 Area 计算 feature (目前用一个 20*20 的矩阵来表征)
    for area in tqdm(areas):
        feat_mat = calc_feat_mat(area)
        cv2.imwrite(f"result/{area.idx}.png", feat_mat)  # fixme: 保存图片, 但重心对齐, 为了思考如何将相似的区域聚类到一起...
        if feat_mat.shape == (20, 20):
            area.feat_mat = feat_mat
            area.feat_mass = np.sum(feat_mat)
        else:
            area.feat_mat = None
            print(f"area_{area.idx} 的尺寸较大, feat_mat 超出 20*20, 不予使用; 况且这说明其自身分辨率也足够大了, 不必尝试与其他合并")

    # 将相似的 Area 聚类到一起
    Cluster = list[Area]                # Cluster 中的 area 应该是相似的, 第一个元素作为 "leader"
    clusters: list[Cluster] = []
    difference_threshold = 0.20         # 差异度阈值: diff ∈ [0, 1], 0: 完全重叠, 1: 完全不重叠.
    shape_ratio_threshold = 0.7         # 边长比例阈值: 仅用于加速, 筛掉明显不匹配的
    mass_ratio_threshold = 0.7          # 灰度面积阈值: 仅用于加速, 筛掉明显不匹配的
    for area in tqdm(areas):
        if area.feat_mat is None:       # 不参与合并的大图案
            clusters.append([area,])    # 作为一个独立的 cluster
            continue
        all_clusters_which_satisfy_threshold: list[(Cluster, float)] = []   # 所有满足阈值的 cluster, 以及其与 area 的差异度, (最后会从中选出最相似的那个 Cluster)
        for cluster in clusters:
            cluster_leader: Area = cluster[0]
            if cluster_leader.feat_mat is None: continue
            if not (shape_ratio_threshold <= (area.h / cluster_leader.h) <= 1 / shape_ratio_threshold): continue
            if not (shape_ratio_threshold <= (area.w / cluster_leader.w) <= 1 / shape_ratio_threshold): continue
            if not (mass_ratio_threshold <= (area.mass / cluster_leader.mass) <= 1 / mass_ratio_threshold): continue
            if (difference := calc_feat_diff(area, cluster_leader)) < difference_threshold:
                all_clusters_which_satisfy_threshold.append((cluster, difference))
        if len(all_clusters_which_satisfy_threshold) == 0:  # 没有找到相似的 cluster, 则新建一个 cluster
            clusters.append([area,])
        else:                                            # 找到了相似的 cluster, 则将 area 加入其中
            all_clusters_which_satisfy_threshold.sort(key=lambda x: x[1])
            most_similar_cluster = all_clusters_which_satisfy_threshold[0][0]
            most_similar_cluster.append(area)

    print(f"对 {len(areas)} 个 Area, 产生了 {len(clusters)} 个 Cluster")

    # 创建 symbol_table, 每个 area 的 symbol_id 指向某个符号, 这个符号由 (grey_mask, cx, cy) 表征 (其中 cx, cy 如果是 centroid() 融合而成的, 那应该都是 int + 0.5 的样子, 因为 2x 超分后 canvas 是偶数边长的, 又因为 centroid() 返回的 area 是 centroid 中心化的).
    symbol_table: list[Symbol] = []

    upscale = 2  # 多帧融合时使用的超采样倍率
    # clusters = [[a] for a in areas]  # debug: 每个 symbol 仅由一个 area 合成
    # 对于每个 Cluster: 将其中包含的 Areas 合并为一个 Symbol, 并让这些 Areas 统统指向这个 Symbol
    for similar_areas in clusters:
        # 根据相似的 Area 制作 Symbol
        grey_mask, cx, cy = centroids_aligned_mean(similar_areas, upscale=upscale)
        symbol_table.append(Symbol(grey_mask=grey_mask, cx=cx, cy=cy))
        for area in similar_areas:
            area.symbol_id = len(symbol_table) - 1

    # 构建一个 4x 大的空白图片, 把每个符号都放到这个图片上.
    new_img = np.zeros(shape=[upscale * grey_img.shape[0], upscale * grey_img.shape[1]])
    for area in areas:
        symbol = symbol_table[area.symbol_id]
        # 计算原 area 的重心在新图片上的位置
        cx_should_be = upscaled_coord(area.cx + area.x, upscale)
        cy_should_be = upscaled_coord(area.cy + area.y, upscale)
        # 计算 symbol 在与 new_img 左上角对齐时, 还需要移动多少距离才能使得重心对齐 c_should_be
        should_move_x = cx_should_be - symbol.cx
        should_move_y = cy_should_be - symbol.cy
        # 将偏移量拆成整数和浮点两部分, 整数部分用索引解决, 浮点部分用 move_subpixel() 解决
        should_move_x_int, should_move_x_f32 = float_split(should_move_x)
        should_move_y_int, should_move_y_f32 = float_split(should_move_y)
        # 浮点移动
        symbol_refined = move_img_subpixel(symbol.grey_mask, should_move_x_f32, should_move_y_f32)
        # 整数移动 (由于可能越界, 目前用 except 忽略潜在的错误)
        try:
            new_img[should_move_y_int: should_move_y_int + symbol_refined.shape[0],
                    should_move_x_int: should_move_x_int + symbol_refined.shape[1]] += symbol_refined
        except Exception as e:
            print(e)

    cv2.imwrite("result/new_img.png", (255-new_img).astype(np.uint8))
    plt.imshow(new_img)
    plt.show()

    # TODO: 把每个符号的矢量勾勒出来, 构建一个 svg, 把每个符号放到 svg 上, 导出 svg.




















