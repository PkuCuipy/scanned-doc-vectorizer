# 2023-03-31(19.59.05)
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import trange, tqdm
from typing import Iterable, Union
import datetime
import svgwrite.path
import potrace_utils

# 区域类
class Area:
    __slots__ = ["idx", "x", "y", "w", "h", "cx", "cy", "mass", "bin_mask", "grey_mask", "symbol_id", "feat_mat", "feat_mass"]
    def __init__(s, *, idx=None, x=None, y=None, w=None, h=None, cx=None, cy=None, mass=None, bin_mask=None, grey_mask=None, symbol_id=None, feat_mat=None, feat_mass=None):
        (s.idx, s.x, s.y, s.w, s.h, s.cx, s.cy, s.mass, s.bin_mask, s.grey_mask, s.symbol_id, s.feat_mat, s.feat_mass) = (idx, x, y, w, h, cx, cy, mass, bin_mask, grey_mask, symbol_id, feat_mat, feat_mass)

# 符号类
class Symbol:
    # 每个符号由 (grey_mask, cx, cy) 表征, 其中 cx, cy 如果是 centroid() 融合而成的, 那应该都是 int + 0.5 的样子,
    # 因为 2x 超分后 canvas 是偶数边长的, 又因为 centroid() 返回的 area 是 centroid 中心化的).
    __slots__ = ["grey_mask", "cx", "cy", "idx", "debug_nr_areas_merged", "debug_area_ids"]
    def __init__(s, *, idx, grey_mask, cx, cy, nr_areas_merged=None, area_ids=None):
        (s.grey_mask, s.cx, s.cy, s.idx, s.debug_nr_areas_merged, s.debug_area_ids) = (grey_mask, cx, cy, idx, nr_areas_merged, area_ids)

# 区域聚簇类:
class Cluster:
    __slots__ = ["areas", "do_not_try_merge", "_feat_mat__sum", "_feat_mass__sum", "_w__sum", "_h__sum", "_mass__sum"]
    def __init__(self, first_area: Area):
        self.areas = [first_area, ]                     # Cluster 中的 area 应该是相似的
        self.do_not_try_merge = (first_area.feat_mat is None)
        if (not self.do_not_try_merge) and (not USE_LEADER_MODE):
            self._w__sum = first_area.w
            self._h__sum = first_area.h
            self._mass__sum = first_area.mass
            self._feat_mat__sum = first_area.feat_mat.copy()    # 这里必须 copy!! 否则会导致后续的 append_area 会修改到 first_area.feat_mat
            self._feat_mass__sum = first_area.feat_mass         # 注意区分 feat_mass 和 mass!! 前者是 sum(feat_mat), 后者是 sum(grey_mask)

    def get_leader(self) -> Area:
        n = len(self.areas)
        if USE_LEADER_MODE:
            return self.areas[0]
        elif n == 1:    # 优化加速
            return self.areas[0]
        else:           # 返回一个虚拟的 areas (仅有 feature 部分非 None), 其 feature 是当前 Cluster 的 areas 的 feature 的平均值
            return Area(w=self._w__sum / n, h=self._h__sum / n, mass=self._mass__sum / n, feat_mat=self._feat_mat__sum / n, feat_mass=self._feat_mass__sum / n)

    def append_area(self, area: Area):
        self.areas.append(area)
        if not USE_LEADER_MODE:
            self._feat_mat__sum += area.feat_mat
            self._w__sum += area.w
            self._h__sum += area.h
            self._mass__sum += area.mass

    def __iter__(self):
        return iter(self.areas)

# 计算矩阵的质心 (比如 [1 1 1; 1 1 1] 的质心是 [0.5, 1])
def calc_centroid(mat: np.ndarray) -> np.ndarray:
    cols_sum = np.sum(mat, axis=1)
    rows_sum = np.sum(mat, axis=0)
    c_i = (cols_sum / np.sum(cols_sum)) @ np.arange(mat.shape[0])
    c_j = (rows_sum / np.sum(rows_sum)) @ np.arange(mat.shape[1])
    return np.array([c_j, c_i])

# 亚像素图片移动
def move_img_subpixel(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    assert abs(dx) < 1.0 and abs(dy) < 1.0, "仅支持亚像素移动"
    h, w = img.shape
    padded = np.pad(img, 1, mode="edge")
    x, y = np.meshgrid(
        np.arange(1, w + 1, dtype="f4") - dx,   # 0 + 1 和 w + 1 是因为 padding
        np.arange(1, h + 1, dtype="f4") - dy    # -dx 和 -dy 是因为图片向右移动相当于采样点左移
    )
    return cv2.remap(padded, x, y, cv2.INTER_LINEAR)

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
    upscaled_padded_a = cv2.resize(padded_a, (0, 0), fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)
    # 超采样之后的 centroid 坐标  (快速计算, 对 NEAREST, BILINEAR 超采样成立, 对 BICUBIC 不成立)
    upscaled_padded_a_cx = upscaled_coord(a.cx + 1, upscale)  # +1 计算 padding 导致的第一次重心挪移; upscaled_coord() 计算 scaling 导致的第二次挪移
    upscaled_padded_a_cy = upscaled_coord(a.cy + 1, upscale)
    # assert np.allclose([upscaled_padded_a_cx, upscaled_padded_a_cy], calc_centroid(upscaled_padded_a))
    # 为了将超采样后的 upscaled_padded_a 和 canvas 的 centroid 对齐, 计算需要移动的量
    canvas_cx = center_of_rect(canvas_w)
    canvas_cy = center_of_rect(canvas_h)
    move_x = canvas_cx - upscaled_padded_a_cx
    move_y = canvas_cy - upscaled_padded_a_cy
    # 将移动量转换为整数和小数部分, 整数部分用于数组级别的移动, 小数部分用于亚像素级别的移动
    move_x_int, move_x_f32 = float_split(move_x)
    move_y_int, move_y_f32 = float_split(move_y)
    # 初始化 area 的 canvas
    canvas = np.zeros((canvas_h, canvas_w), dtype="f4")   # 初始化背景为黑色
    # 整数级别的移动
    canvas[move_y_int : move_y_int + upscaled_padded_a.shape[0], move_x_int : move_x_int + upscaled_padded_a.shape[1]] = upscaled_padded_a
    # 亚像素级别的移动
    canvas = move_img_subpixel(canvas, move_x_f32, move_y_f32)
    return canvas

# 在高分辨率下, 将若干个区域的 centroid 对齐
def centroids_aligned_mean(areas: Iterable[Area], upscale: int, *, canvas_h_at_least_prompt: int = 0, canvas_w_at_least_prompt: int = 0) -> (np.ndarray, float, float):
    # 以下会自行决定合适的 canvas 尺寸以保证能容纳所有重心对齐的 areas,
    # 但用户可以通过 canvas_h_at_least_prompt 和 canvas_w_at_least_prompt 来指定最小尺寸.
    assert upscale > 0 and upscale % 1 == 0
    canvas_h = int(max(max(a.cy, a.h - a.cy) for a in areas) + 3) * 2 * upscale     # +3: 留出余量空间
    canvas_w = int(max(max(a.cx, a.w - a.cx) for a in areas) + 3) * 2 * upscale
    if canvas_h < canvas_h_at_least_prompt: canvas_h = canvas_h_at_least_prompt  # 如果小于用户指定的最小尺寸, 则扩大到这个尺寸
    if canvas_w < canvas_w_at_least_prompt: canvas_w = canvas_w_at_least_prompt  # 如果小于用户指定的最小尺寸, 则扩大到这个尺寸
    all_canvas = np.array([move_img_to_target_canvas_with_centroid_aligned_and_up_scaled(a, upscale, canvas_w, canvas_h) for a in areas])
    return np.mean(all_canvas, axis=0), (canvas_w-1) / 2, (canvas_h-1) / 2

# 传入一个 area, 计算 feature_vector
DOWN_SAMPLE_RATIO = 1
GAUSSIAN_SIGMA = 1.5
FEAT_MAT_SIZE = 60
print("FEAT_MAT_SIZE =", FEAT_MAT_SIZE)
def calc_feat_mat(area: Area) -> Union[np.ndarray, None]:
    feat_mat = centroids_aligned_mean([area, ], upscale=1, canvas_h_at_least_prompt=FEAT_MAT_SIZE, canvas_w_at_least_prompt=FEAT_MAT_SIZE)[0]
    if feat_mat.shape != (FEAT_MAT_SIZE, FEAT_MAT_SIZE):
        return None  # 这个 area 的尺寸很大, 认为分辨率已足够大, 无需合并超分辨率
    feat_mat = cv2.GaussianBlur(feat_mat, (0, 0), sigmaX=GAUSSIAN_SIGMA)    # 高斯模糊, 提高模糊匹配程度
    feat_mat = np.mean(feat_mat.reshape((feat_mat.shape[0] // DOWN_SAMPLE_RATIO, DOWN_SAMPLE_RATIO, feat_mat.shape[1] // DOWN_SAMPLE_RATIO, DOWN_SAMPLE_RATIO)), axis=(1, 3))    # 低通滤波. (cv2.resize 会导致小像素蜜汁消失, 不懂为啥, 总之弃用)
    return feat_mat

# 传入两个 area, 计算他们的 feature 差异度
def calc_feat_diff(a1: Area, a2: Area) -> float:
    # 目前差异度 (difference) 定义为: sum(|A-B|) / (sum(A)+sum(B)), 有种 diff-over-union 的感觉
    return np.sum(np.abs(a1.feat_mat - a2.feat_mat)) / ((a1.feat_mass + a2.feat_mass) / 2)
    # return np.sqrt(np.sum(np.square(a1.feat_mat - a2.feat_mat))) / ((a1.feat_mass + a2.feat_mass) / 2)

# 将一个 pattern 绘制在 canvas 上, 使得 pattern 上的某个坐标点与 canvas 的指定坐标点重合
def place_pattern_on_canvas(pattern: np.ndarray, pattern_cx: float, pattern_cy: float, placement_cx: float, placement_cy: float, canvas: np.ndarray) -> None:
    patH, patW = pattern.shape
    canH, canW = canvas.shape
    # 计算 pattern 在与 canvas 的 (0, 0) 对齐时, 需要再移动多少距离才能使得 [重心] 对齐
    dx = placement_cx - pattern_cx
    dy = placement_cy - pattern_cy
    # 将移动量拆成整数和浮点两部分, 整数部分用索引解决, 浮点部分用 move_subpixel() 解决
    dx_int, dx_f32 = float_split(dx)
    dy_int, dy_f32 = float_split(dy)
    # 浮点移动 (这不会影响 pattern 的 shape)
    pattern_refined = move_img_subpixel(pattern, dx_f32, dy_f32)
    # 整数移动:
    #  1. 计算移动后的 pattern 在 canvas 上的范围
    i_beg, i_end = dy_int, (dy_int + patH)
    j_beg, j_end = dx_int, (dx_int + patW)
    #  2. 这个范围需要修正到 canvas 的边界范围之内 (i.e. 0 ≤ start ≤ end < len)
    i_beg_incr, i_end_decr, j_beg_incr, j_end_decr = 0, 0, 0, 0  # 默认不裁剪, 即默认为 0
    if i_beg < 0:    (i_beg, i_beg_incr) = (0, 0 - i_beg)
    if i_end > canH: (i_end, i_end_decr) = (canH, i_end - canH)
    if j_beg < 0:    (j_beg, j_beg_incr) = (0, 0 - j_beg)
    if j_end > canW: (j_end, j_end_decr) = (canW, j_end - canW)
    #  3. 将 pattern_refined 的指定范围, 覆盖到 canvas 的指定范围
    canvas[i_beg: i_end, j_beg: j_end] += pattern_refined[i_beg_incr: patH - i_end_decr, j_beg_incr: patW - j_end_decr]

# 保存单独一个 <path> 到 .svg 文件
def save_path_to_svg_file(path_elem, svg_size, svg_filename: str) -> None:
    dwg = svgwrite.Drawing(filename=svg_filename, size=svg_size)
    dwg.add(dwg.rect(insert=(0, 0), size=svg_size, fill="white"))   # 垫一个矩形作为背景色
    dwg.add(path_elem)
    dwg.save(pretty=True)

# ======================================================================================================================

if __name__ == "__main__":

    # 创建存放结果的文件夹
    __ROOT_FOLDER = ["/Users/cuipy/NoBackup", "."][0]
    RESULT_FOLDER = f"{__ROOT_FOLDER}/result/result_{datetime.datetime.now().strftime('%Y-%m-%d(%H.%M.%S)')}"
    AREA_FOLDER = f"{RESULT_FOLDER}/areas"
    FEAT_FOLDER = f"{RESULT_FOLDER}/feat_mats"
    SYMBOL_FOLDER = f"{RESULT_FOLDER}/symbols"
    SVG_FOLDER = f"{RESULT_FOLDER}/svg"
    if not os.path.exists(RESULT_FOLDER): os.makedirs(RESULT_FOLDER)
    if not os.path.exists(AREA_FOLDER): os.mkdir(AREA_FOLDER)
    if not os.path.exists(FEAT_FOLDER): os.mkdir(FEAT_FOLDER)
    if not os.path.exists(SYMBOL_FOLDER): os.mkdir(SYMBOL_FOLDER)
    if not os.path.exists(SVG_FOLDER): os.mkdir(SVG_FOLDER)

    # 设置 plt 窗口大小
    plt.rcParams["figure.figsize"] = (13, 9)

    # 以灰度模式读入图片 (注: 本程序中 img 是白底黑字, mask 是黑底白字)
    grey_img = np.array(cv2.imread("data/test_23.png", cv2.IMREAD_GRAYSCALE))
    # grey_img = np.array(cv2.imread("data/scanfile/text-300.tiff", cv2.IMREAD_GRAYSCALE))
    cv2.imwrite(f"{RESULT_FOLDER}/input.png", grey_img)

    # 考虑到在同分辨率下, 灰度图的边缘更加平滑, 所以在二值化前先对灰度图进行放大, 这有利于提取区域(?) fixme: 对二值图像放大是无意义的, 感觉对有抗锯齿的才有意义?
    UPSCALE_GREY_2_BINARY = 1
    grey_img = cv2.resize(grey_img, (0, 0), fx=UPSCALE_GREY_2_BINARY, fy=UPSCALE_GREY_2_BINARY, interpolation=cv2.INTER_LINEAR)
    IMG_H, IMG_W = grey_img.shape
    DARKER_THAN_THAT_BELONGS_TO_MASK = 175
    bin_threshold, bin_mask = cv2.threshold(grey_img, DARKER_THAN_THAT_BELONGS_TO_MASK, 255, cv2.THRESH_BINARY_INV)   # bin_mask 前景为白色 (255)
    cv2.imwrite(f"{RESULT_FOLDER}/input_upscaled.png", grey_img)
    cv2.imwrite(f"{RESULT_FOLDER}/input_binarized.png", 255 - bin_mask)

    # 提取所有的连通区域
    nr_labels, label_map, label_stats, __label_centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)    # fixme: 使用 4-连通, 尽可能缩小每个区域, 似乎更适合英文; 8-连通似乎适合中文
    del bin_mask    # bin_mask 仅用于提取连通区域, 后面不应使用

    # 保存每个连通区域的信息
    areas = []
    for idx in range(1, nr_labels):     # 0 为背景, 忽略
        x, y, w, h, area = label_stats[idx]
        view = (slice(y, y + h), slice(x, x + w))           # x 向右, y 向下, 因此 y 才是 i, x 才是 j!
        _bin_mask = (label_map[view] == idx)
        _grey_mask = np.where(_bin_mask, 255 - grey_img[view], 0)  # 这里假设背景色是 0, 前景是 1~255;  fixme: 被 bin_mask 约束使得遗漏浅色抗锯齿部分
        # _grey_mask = 255 - grey_img[view]  # 这里假设背景色是 0, 前景是 1~255;  fixme: 根据 bbox 框选, 可能多个 bbox 会重复包含某些像素.
        centroid = calc_centroid(_grey_mask)                       # 基于灰度图计算, 此前是基于 bin_mask 计算的 (label_centroids[idx] - [x, y])
        mass = np.sum(_grey_mask)                                  # 基于灰度图计算, 此前是基于 bin_mask 计算的
        areas.append(Area(idx=idx, x=x, y=y, w=w, h=h, cx=centroid[0], cy=centroid[1], mass=mass, bin_mask=_bin_mask, grey_mask=_grey_mask))
    del grey_img        # grey_img 后面不应该使用

    # 绘制每个连通域
    # plt.imshow(label_map, cmap="jet")   # label_map 作为背景色, 绘制 bbox 于其上
    for area in tqdm(areas, desc="绘制每个连通域"):
        # plt.gca().add_patch(plt.Rectangle((area.x - 0.5, area.y - 0.5), area.w, area.h, fill=False, edgecolor="r", linewidth=0.5))    # 绘制 bbox
        # plt.gca().add_patch(plt.Circle((area.x + area.cx, area.y + area.cy), radius=0.5, fill=False, edgecolor="g", linewidth=0.5))   # 绘制 centroid
        cv2.imwrite(f"{AREA_FOLDER}/{area.idx}.png", 255 - area.grey_mask)   # 保存图片
    # plt.show()

    # 为每个 Area 计算 feature_mat
    debug_no_feat_area_indices = [] # debug: 记录没有 feat_mat 的 area 的 idx
    for area in tqdm(areas, desc="为每个 Area 计算 feat_mat"):
        area.feat_mat = calc_feat_mat(area)
        if area.feat_mat is not None:
            area.feat_mass = np.sum(area.feat_mat)
            cv2.imwrite(f"{FEAT_FOLDER}/{area.idx}.png", area.feat_mat)   # 打印 feature_mat
        else:
            debug_no_feat_area_indices.append(area.idx)
    print(f"area: {debug_no_feat_area_indices} (len={len(debug_no_feat_area_indices)}) 的尺寸较大, feat_mat 超出 prompt_size, 不予使用; 况且这说明其自身分辨率也足够大了, 不必尝试与其他合并")

    # 将相似的 Area 聚类到一起
    USE_LEADER_MODE = True          # 如果使用 "leader" 模式, 那么总是返回第一个 area 作为整个 cluster 的代表
    DIFFERENCE_THRESHOLD = 0.20     # 差异度阈值: diff ∈ [0, 1], 0: 完全重叠, 1: 完全不重叠.
    SHAPE_RATIO_THRESHOLD = 0.7     # 边长比例阈值: 仅用于加速, 筛掉明显不匹配的
    MASS_RATIO_THRESHOLD = 0.7      # 灰度面积阈值: 仅用于加速, 筛掉明显不匹配的
    clusters: list[Cluster] = []
    for area in tqdm(areas, desc="将相似的 Area 聚类到一个 Cluster 中"):
        if area.feat_mat is None:           # 不参与合并的大图案
            clusters.append(Cluster(area))  # 作为一个独立的 cluster
            continue
        all_clusters_which_satisfy_threshold: list[tuple[Cluster, float]] = []   # 所有满足阈值的 cluster, 以及其与 area 的差异度, (最后会从中选出最相似的那个 Cluster)
        for cluster in clusters:
            if cluster.do_not_try_merge: continue
            cluster_leader: Area = cluster.get_leader()
            if not (SHAPE_RATIO_THRESHOLD < (area.h / cluster_leader.h) < 1 / SHAPE_RATIO_THRESHOLD): continue        # 高度差太多, 直接否决
            if not (SHAPE_RATIO_THRESHOLD < (area.w / cluster_leader.w) < 1 / SHAPE_RATIO_THRESHOLD): continue        # 宽度差太多, 直接否决
            if not (MASS_RATIO_THRESHOLD < (area.mass / cluster_leader.mass) < 1 / MASS_RATIO_THRESHOLD): continue    # 面积差太多, 直接否决
            if (difference := calc_feat_diff(area, cluster_leader)) < DIFFERENCE_THRESHOLD:     # 差异度小于阈值, 加入
                all_clusters_which_satisfy_threshold.append((cluster, difference))
        if len(all_clusters_which_satisfy_threshold) == 0:  # 没有找到相似的 cluster, 则新建一个 cluster
            clusters.append(Cluster(area))
        else:                                            # 找到了相似的 cluster, 则将 area 加入其中
            all_clusters_which_satisfy_threshold.sort(key=lambda x: x[1])
            most_similar_cluster = all_clusters_which_satisfy_threshold[0][0]
            most_similar_cluster.append_area(area)
    print(f"对 {len(areas)} 个 Area, 产生了 {len(clusters)} 个 Cluster")

    # 融合每个 Cluster 中的 Areas, 生成 Symbol, 并让这些 Areas 都指向这个 Symbol
    symbol_table: list[Symbol] = []     # 下标 [i] 的 Symbol 必有 idx == i
    UPSCALE_AREA_2_SYMBOL = 2           # 多个 Area 融合成 Symbol 前使用的超采样倍率
    for clu in clusters:
        # 根据相似的 Area 制作 Symbol
        new_symbol_idx = len(symbol_table)
        _grey_mask, cx, cy = centroids_aligned_mean(clu.areas, upscale=UPSCALE_AREA_2_SYMBOL)
        symbol_table.append(Symbol(idx=new_symbol_idx, grey_mask=_grey_mask, cx=cx, cy=cy, nr_areas_merged=len(clu.areas), area_ids=[area.idx for area in clu.areas]))
        for area in clu.areas:
            area.symbol_id = new_symbol_idx

    # Debug: 输出所有的 symbol
    [cv2.imwrite(f"{SYMBOL_FOLDER}/nr={sym.debug_nr_areas_merged}_areas={sym.debug_area_ids[:3]}_sid={sym.idx}.png", sym.grey_mask) for sym in symbol_table]

    # 构建一个 upscale 的空白画布, 把每个符号都放到这个图片上.
    canvas = np.zeros(shape=[UPSCALE_AREA_2_SYMBOL * IMG_H, UPSCALE_AREA_2_SYMBOL * IMG_W])
    # canvas += (BG_COLOR := 23)
    for area in tqdm(areas, desc="将所有的 Symbol 绘制到新的图片上"):
        symbol = symbol_table[area.symbol_id]
        # 计算原 area 的重心在新图片上的位置
        cx_should_be = upscaled_coord(area.cx + area.x, UPSCALE_AREA_2_SYMBOL)
        cy_should_be = upscaled_coord(area.cy + area.y, UPSCALE_AREA_2_SYMBOL)
        # 将 Symbol 绘制在新图片的指定坐标处
        place_pattern_on_canvas(symbol.grey_mask, symbol.cx, symbol.cy, cx_should_be, cy_should_be, canvas)

    # 以位图保存 SR 结果
    SR_img = (255 - canvas).clip(0, 255)
    cv2.imwrite(f"{RESULT_FOLDER}/SR.png", SR_img)
    cv2.imwrite(f"{RESULT_FOLDER}/SR_binarize.png", cv2.threshold(cv2.resize(SR_img, (0, 0), fx=UPSCALE_GREY_2_BINARY, fy=UPSCALE_GREY_2_BINARY, interpolation=cv2.INTER_LINEAR), 127, 255, cv2.THRESH_BINARY)[1])

    # 把每个符号的矢量勾勒出来, 构建一个 svg, 把每个符号放到 svg 上, 导出 svg.
    drawing = svgwrite.Drawing(f"{RESULT_FOLDER}/output.svg", size=("100%", "100%"))  # 构建一个 <svg> 画布
    drawing.elements = []                                                                               # 清空该 .svg 文件的原有内容
    drawing.viewbox(-0.5, -0.5, SR_img.shape[1] + 0.5, SR_img.shape[0] + 0.5)
    drawing.add(drawing.rect(insert=(0, 0), size=[SR_img.shape[1], SR_img.shape[0]], fill="white"))     # 垫一个矩形作为背景色

    # 为每个 Symbol 构建矢量表示, 然后注册为 <symbol>
    USE_POTRACE = True
    debug_no_cmd_symbol_indices = []
    for sym in tqdm(symbol_table):
        sym_grey = sym.grey_mask    # 该符号的灰度图, 其轮廓将基于该图描绘
        if USE_POTRACE:             # 使用 potrace 提取轮廓
            # 使用 potrace 提取轮廓, 得到 <g> 的各属性
            g_elem_info = potrace_utils.mat_to_g_elem(255 - sym_grey)   # potrace 的逻辑是 ｢黑色｣ 为前景, 因此这里需要反色
            if g_elem_info["d"] == "":
                debug_no_cmd_symbol_indices.append(sym.idx)
                continue
            g_elem = drawing.g(transform=g_elem_info["transform"], fill="#000000",stroke="none")
            g_elem.add(drawing.path(d=g_elem_info["d"]))
            # 注册 <symbol>
            symbol_elem = drawing.symbol(id=f"symbol_{sym.idx}")  # 注意这里不加 `#` 前缀!
            symbol_elem.add(g_elem)
            # 将 <symbol> 放到 <svg> 上
            drawing.add(symbol_elem)
        else:                       # 使用 cv2.findContours() 提取轮廓
            sym_bin = cv2.threshold(cv2.resize(sym_grey, (0, 0), fx=UPSCALE_GREY_2_BINARY, fy=UPSCALE_GREY_2_BINARY, interpolation=cv2.INTER_LINEAR), 127, 255, cv2.THRESH_BINARY)[1]
            cx, cy = upscaled_coord(sym.cx, 2), upscaled_coord(sym.cy, 2)
            contours, hierarchy = cv2.findContours(image=sym_bin.astype("u1"), mode=cv2.RETR_CCOMP, # RETR_CCOMP: 提取两层级的轮廓 (0 层级是外轮廓, 1 层级是内轮廓)
                                                   method=cv2.CHAIN_APPROX_TC89_KCOS)               # method 是减少顶点数量选用的算法
            # 构建绘制指令序列
            path_cmds: list[str] = []
            for contour in contours:                        # 比如 `器` 字包含 5 个 contour
                points = contour.squeeze(1)                 # dim=1 恒为 1, 这是来自 C++ 的兼容性设计
                points = points / UPSCALE_GREY_2_BINARY     # 由于是在放大的图片上提取的轮廓, 所以需要缩小回来
                path_cmds.append("M")                       # M x0,y0  L x1,y1  L x2,y2  ...  Z
                for i, point in enumerate(points):
                    path_cmds.append(f"{point[0]},{point[1]}")
                    if i != len(points) - 1:
                        path_cmds.append("L")
                path_cmds.append("Z")

            # 如果有绘制指令, 则构建 <path> 并注册为 <symbol>
            if path_cmds:
                # 构建 <path>
                path_elem = svgwrite.path.Path(d=" ".join(path_cmds), fill='black')
                save_path_to_svg_file(path_elem, [sym.grey_mask.shape[1], sym.grey_mask.shape[0]], f"{SVG_FOLDER}/symbol_{sym.idx}.svg")
                # 注册 <symbol>
                symbol_elem = drawing.symbol(id=f"symbol_{sym.idx}")    # 注意这里不加 `#` 前缀!
                symbol_elem.add(path_elem)
                # 将 <symbol> 放到 <svg> 上
                drawing.add(symbol_elem)
            else:
                debug_no_cmd_symbol_indices.append(sym.idx)
    print(f"Symbol: {debug_no_cmd_symbol_indices} (len={len(debug_no_cmd_symbol_indices)}) 不存在绘制指令!")

    # 对于每个 Area, 使用 <use> 引用其对应的 <symbol>
    for area in tqdm(areas, desc="将每个 Area 用 <use> 引用其对应的 <symbol>"):
        sid: int = area.symbol_id
        sym: Symbol = symbol_table[sid]
        insert_x = upscaled_coord(area.x + area.cx, UPSCALE_AREA_2_SYMBOL) - sym.cx
        insert_y = upscaled_coord(area.y + area.cy, UPSCALE_AREA_2_SYMBOL) - sym.cy
        use_elem = drawing.use(f"#symbol_{area.symbol_id}", insert=(insert_x, insert_y))     # 注意这里要加 `#` 前缀!
        drawing.add(use_elem)

    # 保存 SVG
    drawing.save(pretty=True)


