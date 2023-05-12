# 2023-05-03

from __future__ import annotations
import fitz
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from itertools import product
import svgwrite
from scipy.signal import convolve2d
from pathlib import Path
import random
import multiprocessing
import functools


# 传入一个 .svg 文件, 进行网格密集采样  (注: 其实我们完全不 care SVG 的矢量表示, 只是因为这玩意可以超高精度采样, 比如如果有 8K 的字符光栅图像, 那理论上也是 ok 的)
def svg_to_grid(svg_file_path: str | Path, n_blocks_vert: int, n_blocks_horiz: int, n_subdiv: int) -> np.ndarray:
    # n_blocks: 网格边上的 [大方块] 数
    # n_subdiv: 每个 [大方块] 的 [小方块] 细分数 (最小为 1 即不细分)
    nr_padding_blocks = 1
    nr_inner_blocks_vert = n_blocks_vert - 2 * nr_padding_blocks
    nr_inner_blocks_horiz = n_blocks_horiz - 2 * nr_padding_blocks
    grid_height = n_blocks_vert * n_subdiv + 1
    grid_width = n_blocks_horiz * n_subdiv + 1
    inner_grid_height = nr_inner_blocks_vert * n_subdiv + 1
    inner_grid_width = nr_inner_blocks_horiz * n_subdiv + 1
    grid: np.ndarray = np.ones([grid_height, grid_width]) * 255.0
    # 读入 svg
    svg_bytes = open(svg_file_path, "rb").read()
    doc = fitz.Document("svg", svg_bytes)
    if (zoom := 1.0) != 1.0:                    # 中心缩放, 比如 zoom = 1.5 可以解决原 svg 的 padding 太大的问题
        pdf_bytes = doc.convert_to_pdf()        # svg 在 fitz 下不支持 set_mediabox(), 要先转为 pdf
        doc = fitz.Document("pdf", pdf_bytes)
        page = doc.load_page(0)
        cx, cy = page.rect.width / 2, page.rect.height / 2
        page.set_mediabox(fitz.Rect(cx * (1 - 1 / zoom), cy * (1 - 1 / zoom), cx * (1 + 1 / zoom), cy * (1 + 1 / zoom)))
    page = doc.load_page(0)
    # 根据 svg 的长宽, 计算一个缩放比例, 使得恰好撑满 inner_box
    svg_width = page.rect.width
    svg_height = page.rect.height
    svg_aspect_ratio = svg_width / svg_height
    target_aspect_ratio = inner_grid_width / inner_grid_height
    scale = (inner_grid_height / svg_height) if (svg_aspect_ratio < target_aspect_ratio) else (inner_grid_width / svg_width) # svg 相对瘦高, 则高度是约束; 否则宽度是约束
    # 调用 get_pixmap() 得到的是像素值, 但在 subdiv_per_block 很大的时候, 可以近似为格点值.
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace="gray")
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    mat = np.asarray(img)
    # 将 mat[, ] 放置在 grid[, ] 的中心
    grid_center_horiz = grid_width // 2 + 1
    grid_center_vert = grid_height // 2 + 1
    mat_center_horiz = mat.shape[1] // 2 + 1
    mat_center_vert = mat.shape[0] // 2 + 1
    bias_horiz = grid_center_horiz - mat_center_horiz   # >= 0
    bias_vert = grid_center_vert - mat_center_vert      # >= 0
    grid[bias_vert : bias_vert+mat.shape[0], bias_horiz : bias_horiz+mat.shape[1]] = mat
    # 使得 grid ∈ [-1, 1], 且 [+] 为内部, [-] 为外部
    grid = -(grid - 127.5) / 127.5
    return grid


# 以 block 为单位, 上下左右平移传入的 grid
def random_shifted_grid(grid:np.ndarray, block_size: int, shift_x: float = None, shift_y: float = None):
    # 移动最多 0.5 个 block 即可, 因为 CNN 有平移不变性
    if shift_x is None: shift_x = np.random.uniform(-0.5, 0.5)
    if shift_y is None: shift_y = np.random.uniform(-0.5, 0.5)
    return np.roll(grid, shift=[int(shift_x * block_size), int(shift_y * block_size)], axis=(0, 1))


# 根据 [四角正负类型] 初步映射到 Case 编号
cornerType_2_caseNum = {
    (0, 0, 0, 0): 0,
    (1, 1, 1, 1): 1,
    (1, 0, 0, 0): 2,
    (0, 1, 0, 0): 3,
    (0, 0, 1, 0): 4,
    (0, 0, 0, 1): 5,
    (0, 1, 1, 1): 6,
    (1, 0, 1, 1): 7,
    (1, 1, 0, 1): 8,
    (1, 1, 1, 0): 9,
    (1, 1, 0, 0): 10,
    (0, 0, 1, 1): 11,
    (1, 0, 0, 1): 12,
    (0, 1, 1, 0): 13,
    (1, 0, 1, 0, False): 14,
    (1, 0, 1, 0, True): 15,
    (0, 1, 0, 1, False): 16,
    (0, 1, 0, 1, True): 17,
    (1, 0, 1, 0): 1415,
    (0, 1, 0, 1): 1617,
}


# Marching Square 的 18 种情形
class Case:
    __slots__ = ("no", "v1", "tu", "e1", "e2", "e3", "e4", "f1", "f2", "f3", "f4")  # tu: 正结点是否构成 tunnel, 仅在 14 15 16 17 时有意义
    def __init__(self, *, no: int = -1, v1: bool = None, tu: bool = None, e1=None, e2=None, e3=None, e4=None, f1=None, f2=None, f3=None, f4=None):
        self.no, self.v1, self.tu, self.e1, self.e2, self.e3, self.e4, self.f1, self.f2, self.f3, self.f4 = no, v1, tu, e1, e2, e3, e4, f1, f2, f3, f4
    def __repr__(self):
        return f"Case(no={self.no}, v1={self.v1}, tu={self.tu}, \n\t e1={self.e1}, e2={self.e2}, e3={self.e3}, e4={self.e4}, \n\t f1={self.f1}, \n\t f2={self.f2}, \n\t f3={self.f3}, \n\t f4={self.f4})"
    def __float__(self):
        return 0.0


# 计算二维向量的长度的平方. 参数 x 是 (N,2) 的, 返回值是 (N,) 的.
def length_square(x: torch.Tensor) -> torch.Tensor:
    x = x.view(-1, 2)
    return (x ** 2 + 0.001).sum(dim=1)   # 加 0.001 也防止梯度 NaN


# 给定一维数组 edge, 计算 +/- 交界处的位置 e ∈ [0, 1], 返回一个二元向量
def calc_e(i: int, edge: np.ndarray) -> np.ndarray:
    div_left_idx = np.argwhere((edge[0] * edge) > 0).ravel()[-1]    # 找到最后一个和 edge[0] 同号的元素, 即为分界点
    div_right_idx = div_left_idx + 1
    div_left = edge[div_left_idx]
    div_right = edge[div_right_idx]
    left_weight = -div_right / (div_left - div_right)
    right_weight = div_left / (div_left - div_right)
    e = (left_weight * div_left_idx + right_weight * div_right_idx) / (edge.shape[0] - 1)
    if i == 1:   return np.array([0.0, e], dtype="f4")
    elif i == 2: return np.array([e, 1.0], dtype="f4")
    elif i == 3: return np.array([1.0, e], dtype="f4")
    elif i == 4: return np.array([e, 0.0], dtype="f4")


# 计算二维平面内 [点集 C] 到 [线段 AB] 的距离
def dist_pts_to_seg(C: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    AB = B - A
    AC = C - A
    lenAB2 = length_square(AB)
    p = ((AC @ AB) / lenAB2).view(-1, 1).clamp(0.001, 0.999).detach()  # clamp 不用 (0, 1) 是防止梯度 NaN
    AH = p * AB
    CH = AH - AC
    lenCH = torch.sqrt(length_square(CH))   # 加 0.001 也是防止梯度 NaN
    return lenCH


# 以下两个优化函数的超参数
class OptimCfg:
    OPT = torch.optim.SGD   # Adam 不知道为啥特别偏爱让点的坐标趋于 1, 而且收敛特别慢; SGD 奇迹般地表现不错!
    LR = 0.03
    MAX_ITER = 100
    EARLY_BRK_THR = 1e-3    # 基于 loss 的优化早停阈值
    REG_COEF = 0.03         # 让边的长度尽可能小的正则项系数


# 优化 [点 p], 使得 [折线段 A——p——B] 拟合 [点集 points]
def optimized_single_p(*, A, B, points: np.ndarray) -> np.ndarray:
    # fixme: 其实觉得这里用梯度下降法可能不是最好的方案, 因为不好控制迭代步数, 收敛偏慢. \
    #        比如一个改进算法的 idea 是: 首先将每个点归入最近的线段,
    #        对于归入线段 A-P 的点, 计算第一奇异向量得到一条直线 l1; 对于归入线段 P-B 的点, 计算第一奇异向量得到另一条直线 l2. \
    #        然后求 l1 和 l2 的交点, 作为新的 P. 一直迭代下去直到收敛. 这样可能会比现在的要快. \
    A = torch.tensor(A, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)
    points = torch.tensor(points, dtype=torch.float32)
    p = (p_init := 0.5 * (A + B)).requires_grad_(True)
    optimizer = OptimCfg.OPT([p], lr=OptimCfg.LR)
    last_loss = 0
    for i in range(OptimCfg.MAX_ITER):
        optimizer.zero_grad()
        loss = torch.mean(torch.minimum(
            dist_pts_to_seg(points, A, p),
            dist_pts_to_seg(points, B, p)
        )) + OptimCfg.REG_COEF * (length_square(A - p) + length_square(p - B))
        loss.backward()
        optimizer.step()
        # print(loss)
        if abs((last_loss - loss) / loss) < OptimCfg.EARLY_BRK_THR:
            break
        last_loss = loss
    return p.detach().clamp(0.001, 0.999).numpy()


# 优化 [点 p1 和 p2], 使得 [折线段 A——p1——p2——B] 拟合 [点集 points]
def optimized_p1_and_p2(*, A, B, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = torch.tensor(A, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)
    points = torch.tensor(points, dtype=torch.float32)
    p1 = (p1_init := A * 0.67 + B * 0.33).requires_grad_(True)
    p2 = (p2_init := A * 0.33 + B * 0.67).requires_grad_(True)
    optimizer = OptimCfg.OPT([p1, p2], lr=OptimCfg.LR)
    last_loss = 0
    for i in range(OptimCfg.MAX_ITER):
        optimizer.zero_grad()
        loss = torch.mean(torch.minimum(torch.minimum(
            dist_pts_to_seg(points, A, p1),
            dist_pts_to_seg(points, p1, p2)),
            dist_pts_to_seg(points, p2, B))) + OptimCfg.REG_COEF * (length_square(A - p1) + length_square(p1 - p2) + length_square(p2 - B))
        loss.backward()
        optimizer.step()
        # print(loss)
        if abs((last_loss - loss) / loss) < OptimCfg.EARLY_BRK_THR:
            break
        last_loss = loss
    return p1.detach().clamp(0.001, 0.999).numpy(), p2.detach().clamp(0.001, 0.999).numpy()


# 拉普拉斯边缘提取
def laplacian_edge_detector(mat: np.ndarray, *, laplacian_kernel=np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])) -> np.ndarray:
    return convolve2d(mat, laplacian_kernel, boundary="symm", mode="same")


# 返回 label_mat 中 label 表征的区域对应的边界点的坐标 ∈ [0,1]×[0,1]
def edge_points(label_mat: np.ndarray, label: int) -> np.ndarray:
    area = (label_mat == label)
    edge = laplacian_edge_detector(area)
    ijs = np.argwhere(edge)
    xys = ijs / (label_mat.shape[0] - 1)  # xy ∈ [0,1]×[0,1]
    return xys


# 传入一个子矩阵的网格点, 返回对应的 case
def block_to_case(block_grid: np.ndarray) -> Case:
    top_edge = block_grid[0, :]
    left_edge = block_grid[:, 0]
    right_edge = block_grid[:, -1]
    bottom_edge = block_grid[-1, :]

    # 利用 [四角] 的值初步判断 case number
    corner_type = (block_grid[0, 0] > 0, block_grid[0, -1] > 0, block_grid[-1, -1] > 0, block_grid[-1, 0] > 0)
    case_num = cornerType_2_caseNum[corner_type]

    # 利用 [连通域] 判别 case 的合法性, 以及区分 14 ↔ 15 和 16 ↔ 17. (注: 只用 pos_island 是不够的, 因为不能排除其内部中空的情形. 比如一个全 1 方阵, 中心有个 0, 这个是非法 case, 但如果只检查 positive 下的连通性, 则会被错误地归为 case 1.)
    nr_labels_pos, pos_islands = cv2.connectedComponents((positive := (block_grid > 0).astype("u1")), connectivity=4)  # nr_labels_pos := [正岛屿]个数 + 1(背景)
    nr_labels_neg, neg_islands = cv2.connectedComponents(1 - positive, connectivity=4)                          # nr_labels_neg := [负岛屿]个数 + 1(背景)
    nr_pos_islands, nr_neg_islands = nr_labels_pos - 1, nr_labels_neg - 1
    if case_num == 0:
        if not (nr_pos_islands == 0):
            case_num = None
    elif case_num == 1:
        if not (nr_pos_islands == 1 and nr_neg_islands == 0):
            case_num = None
    elif case_num in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}:
        if not (nr_pos_islands == 1 and nr_neg_islands == 1):
            case_num = None
    elif case_num == 1415:
        if nr_pos_islands == 2 and nr_neg_islands == 1:
            case_num = 14
        elif nr_pos_islands == 1 and nr_neg_islands == 2:
            case_num = 15
        else:
            case_num = None
    elif case_num == 1617:
        if nr_pos_islands == 2 and nr_neg_islands == 1:
            case_num = 16
        elif nr_pos_islands == 1 and nr_neg_islands == 2:
            case_num = 17
        else:
            case_num = None

    # 计算每个方块的 int(no), bool(v1, tu), float(e1, e4, f1, f2, f3, f4)
    if case_num is None:
        return Case(no=-1)
    elif case_num == 0:
        v1 = False
        return Case(no=0, v1=v1)
    elif case_num == 1:
        v1 = True
        return Case(no=1, v1=v1)
    elif case_num == 2:
        v1 = True
        e4 = calc_e(4, left_edge)
        e1 = calc_e(1, top_edge)
        f1 = optimized_single_p(A=e4, B=e1, points=edge_points(pos_islands, pos_islands[0, 0]))
        return Case(no=2, v1=v1, e4=e4, e1=e1, f1=f1)
    elif case_num == 3:
        v1 = False
        e1 = calc_e(1, top_edge)
        e2 = calc_e(2, right_edge)
        f2 = optimized_single_p(A=e1, B=e2, points=edge_points(pos_islands, pos_islands[0, -1]))
        return Case(no=3, v1=v1, e1=e1, e2=e2, f2=f2)
    elif case_num == 4:
        v1 = False
        e2 = calc_e(2, right_edge)
        e3 = calc_e(3, bottom_edge)
        f3 = optimized_single_p(A=e2, B=e3, points=edge_points(pos_islands, pos_islands[-1, -1]))
        return Case(no=4, v1=v1, e2=e2, e3=e3, f3=f3)
    elif case_num == 5:
        v1 = False
        e3 = calc_e(3, bottom_edge)
        e4 = calc_e(4, left_edge)
        f4 = optimized_single_p(A=e3, B=e4, points=edge_points(pos_islands, pos_islands[-1, 0]))
        return Case(no=5, v1=v1, e3=e3, e4=e4, f4=f4)
    elif case_num == 6:
        v1 = False
        e4 = calc_e(4, left_edge)
        e1 = calc_e(1, top_edge)
        f1 = optimized_single_p(A=e4, B=e1, points=edge_points(neg_islands, neg_islands[0, 0]))
        return Case(no=6, v1=v1, e4=e4, e1=e1, f1=f1)
    elif case_num == 7:
        v1 = True
        e1 = calc_e(1, top_edge)
        e2 = calc_e(2, right_edge)
        f2 = optimized_single_p(A=e1, B=e2, points=edge_points(neg_islands, neg_islands[0, -1]))
        return Case(no=7, v1=v1, e1=e1, e2=e2, f2=f2)
    elif case_num == 8:
        v1 = True
        e2 = calc_e(2, right_edge)
        e3 = calc_e(3, bottom_edge)
        f3 = optimized_single_p(A=e2, B=e3, points=edge_points(neg_islands, neg_islands[-1, -1]))
        return Case(no=8, v1=v1, e2=e2, e3=e3, f3=f3)
    elif case_num == 9:
        v1 = True
        e3 = calc_e(3, bottom_edge)
        e4 = calc_e(4, left_edge)
        f4 = optimized_single_p(A=e3, B=e4, points=edge_points(neg_islands, neg_islands[-1, 0]))
        return Case(no=9, v1=v1, e3=e3, e4=e4, f4=f4)
    elif case_num == 10:
        v1 = True
        e4 = calc_e(4, left_edge)
        e2 = calc_e(2, right_edge)
        f4, f3 = optimized_p1_and_p2(A=e4, B=e2, points=edge_points(pos_islands, pos_islands[0, 0]))
        return Case(no=10, v1=v1, e4=e4, e2=e2, f4=f4, f3=f3)
    elif case_num == 11:
        v1 = False
        e4 = calc_e(4, left_edge)
        e2 = calc_e(2, right_edge)
        f1, f2 = optimized_p1_and_p2(A=e4, B=e2, points=edge_points(pos_islands, pos_islands[-1, -1]))
        return Case(no=11, v1=v1, e4=e4, e2=e2, f1=f1, f2=f2)
    elif case_num == 12:
        v1 = True
        e1 = calc_e(1, top_edge)
        e3 = calc_e(3, bottom_edge)
        f2, f3 = optimized_p1_and_p2(A=e1, B=e3, points=edge_points(pos_islands, pos_islands[0, 0]))
        return Case(no=12, v1=v1, e1=e1, e3=e3, f2=f2, f3=f3)
    elif case_num == 13:
        v1 = False
        e1 = calc_e(1, top_edge)
        e3 = calc_e(3, bottom_edge)
        f1, f4 = optimized_p1_and_p2(A=e1, B=e3, points=edge_points(pos_islands, pos_islands[-1, -1]))
        return Case(no=13, v1=v1, e1=e1, e3=e3, f1=f1, f4=f4)
    elif case_num == 14:
        v1 = True
        tu = False
        e1, e2, e3, e4 = calc_e(1, top_edge), calc_e(2, right_edge), calc_e(3, bottom_edge), calc_e(4, left_edge)
        f1 = optimized_single_p(A=e4, B=e1, points=edge_points(pos_islands, pos_islands[0, 0]))
        f3 = optimized_single_p(A=e2, B=e3, points=edge_points(pos_islands, pos_islands[-1, -1]))
        return Case(no=14, tu=tu, v1=v1, e1=e1, e2=e2, e3=e3, e4=e4, f1=f1, f3=f3)
    elif case_num == 15:
        v1 = True
        tu = True
        e1, e2, e3, e4 = calc_e(1, top_edge), calc_e(2, right_edge), calc_e(3, bottom_edge), calc_e(4, left_edge)
        f2 = optimized_single_p(A=e1, B=e2, points=edge_points(neg_islands, neg_islands[0, -1]))
        f4 = optimized_single_p(A=e3, B=e4, points=edge_points(neg_islands, neg_islands[-1, 0]))
        return Case(no=15, v1=v1, tu=tu, e1=e1, e2=e2, e3=e3, e4=e4, f2=f2, f4=f4)
    elif case_num == 16:
        v1 = False
        tu = False
        e1, e2, e3, e4 = calc_e(1, top_edge), calc_e(2, right_edge), calc_e(3, bottom_edge), calc_e(4, left_edge)
        f2 = optimized_single_p(A=e1, B=e2, points=edge_points(pos_islands, pos_islands[0, -1]))
        f4 = optimized_single_p(A=e3, B=e4, points=edge_points(pos_islands, pos_islands[-1, 0]))
        return Case(no=16, v1=v1, tu=tu, e1=e1, e2=e2, e3=e3, e4=e4, f2=f2, f4=f4)
    elif case_num == 17:
        v1 = False
        tu = True
        e1, e2, e3, e4 = calc_e(1, top_edge), calc_e(2, right_edge), calc_e(3, bottom_edge), calc_e(4, left_edge)
        f1 = optimized_single_p(A=e4, B=e1, points=edge_points(neg_islands, neg_islands[0, 0]))
        f3 = optimized_single_p(A=e2, B=e3, points=edge_points(neg_islands, neg_islands[-1, -1]))
        return Case(no=17, v1=v1, tu=tu, e1=e1, e2=e2, e3=e3, e4=e4, f1=f1, f3=f3)


# 传入整个 Grid, 返回以 Case 为元素的二维矩阵
def grid_to_case_mat(grid: np.ndarray, nr_blocks_vert: int, nr_blocks_horiz: int, subdiv_per_block: int) -> np.ndarray:
    case_mat = np.zeros(shape=(nr_blocks_vert, nr_blocks_horiz), dtype=Case)
    # for i, j in tqdm(product(range(case_mat.shape[0]), range(case_mat.shape[1])), total=case_mat.size, desc="grid -> case_mat", leave=False):
    for i, j in product(range(case_mat.shape[0]), range(case_mat.shape[1])):
        block_grid = grid[i * subdiv_per_block : (i + 1) * subdiv_per_block + 1, j * subdiv_per_block:(j + 1) * subdiv_per_block + 1]  # 取出当前子矩阵对应的 sub_grid
        case_mat[i, j] = block_to_case(block_grid)
    return case_mat


# 传入 case_mat, 导出到 .svg 文件
def case_mat_to_svg(case_mat: np.ndarray, svg_save_path: str | Path, *, draw_nodes=True, draw_grids=True) -> None:
    dwg = svgwrite.Drawing(filename=svg_save_path, size=("100%", "100%"), viewBox=("0 0 %d %d" % (case_mat.shape[1] + 1, case_mat.shape[0] + 1)))
    polylines: list[list[np.ndarray]] = []
    circles: list[np.ndarray] = []
    for i, j in product(range(case_mat.shape[0]), range(case_mat.shape[1])):
        c = case_mat[i, j]
        ij = np.array([i, j])
        if c.no == -1: continue
        elif c.no == 0: continue
        elif c.no == 1: continue
        elif c.no == 2: polylines.append([c.e4+ij, c.f1+ij, c.e1+ij]); circles.append(c.f1+ij)
        elif c.no == 3: polylines.append([c.e1+ij, c.f2+ij, c.e2+ij]); circles.append(c.f2+ij)
        elif c.no == 4: polylines.append([c.e2+ij, c.f3+ij, c.e3+ij]); circles.append(c.f3+ij)
        elif c.no == 5: polylines.append([c.e3+ij, c.f4+ij, c.e4+ij]); circles.append(c.f4+ij)
        elif c.no == 6: polylines.append([c.e4+ij, c.f1+ij, c.e1+ij]); circles.append(c.f1+ij)
        elif c.no == 7: polylines.append([c.e1+ij, c.f2+ij, c.e2+ij]); circles.append(c.f2+ij)
        elif c.no == 8: polylines.append([c.e2+ij, c.f3+ij, c.e3+ij]); circles.append(c.f3+ij)
        elif c.no == 9: polylines.append([c.e3+ij, c.f4+ij, c.e4+ij]); circles.append(c.f4+ij)
        elif c.no == 10: polylines.append([c.e4+ij, c.f4+ij, c.f3+ij, c.e2+ij]); circles.extend([c.f4+ij, c.f3+ij])
        elif c.no == 11: polylines.append([c.e4+ij, c.f1+ij, c.f2+ij, c.e2+ij]); circles.extend([c.f1+ij, c.f2+ij])
        elif c.no == 12: polylines.append([c.e1+ij, c.f2+ij, c.f3+ij, c.e3+ij]); circles.extend([c.f2+ij, c.f3+ij])
        elif c.no == 13: polylines.append([c.e1+ij, c.f1+ij, c.f4+ij, c.e3+ij]); circles.extend([c.f1+ij, c.f4+ij])
        elif c.no == 14: polylines.append([c.e4+ij, c.f1+ij, c.e1+ij]); polylines.append([c.e3+ij, c.f3+ij, c.e2+ij]); circles.extend([c.f1+ij, c.f3+ij])
        elif c.no == 15: polylines.append([c.e4+ij, c.f4+ij, c.e3+ij]); polylines.append([c.e1+ij, c.f2+ij, c.e2+ij]); circles.extend([c.f4+ij, c.f2+ij])
        elif c.no == 16: polylines.append([c.e4+ij, c.f4+ij, c.e3+ij]); polylines.append([c.e1+ij, c.f2+ij, c.e2+ij]); circles.extend([c.f4+ij, c.f2+ij])
        elif c.no == 17: polylines.append([c.e4+ij, c.f1+ij, c.e1+ij]); polylines.append([c.e3+ij, c.f3+ij, c.e2+ij]); circles.extend([c.f1+ij, c.f3+ij])
    # 绘制 svg 时要注意将 x, y 坐标反过来以适应 svg 的坐标系
    dwg.add(dwg.rect(insert=(0, 0), size=(case_mat.shape[1], case_mat.shape[0]), fill="#fff"))      # 垫一个白底背景
    for polyline in polylines:
        points = [(float(p[1].round(3)), float(p[0].round(3))) for p in polyline]
        dwg.add(dwg.polyline(points=points, stroke="black", fill="none", stroke_width="0.05"))
    if draw_nodes:
        for circle in circles:
            dwg.add(dwg.circle(center=(float(circle[1].round(3)), float(circle[0].round(3))), r="0.05", stroke="red", fill="yellow", stroke_width="0.01"))
    if draw_grids:
        for i, j in product(range(case_mat.shape[0]), range(case_mat.shape[1])):
            dwg.add(dwg.rect(insert=(j, i), size=(1, 1), fill="none", stroke="#ddd", stroke_width="0.02"))  # 网格线
    dwg.save(pretty=True)


# 传入 case_mat, 返回紧凑表示 (注: 相邻的共享节点不重复存储, 舍弃最后一行和最后一列. 用于 CNN 训练)
def case_mat_to_compact(case_mat: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bool_part = np.zeros(shape=(*case_mat.shape, 2), dtype=bool)    # M × N × 2
    bool_mask = np.zeros(shape=(*case_mat.shape, 2), dtype=bool)    # M × N × 2
    float_part = np.zeros(shape=(*case_mat.shape, 10), dtype="f4")  # M × N × 10
    float_part += [0.5,0.5, 0.3,0.3, 0.3,0.7, 0.7,0.7, 0.7,0.3]     # 默认初始值
    float_mask = np.zeros(shape=(*case_mat.shape, 10), dtype=bool)  # M × N × 10
    for i, j in product(range(case_mat.shape[0]), range(case_mat.shape[1])):
        case = case_mat[i, j]
        if case.v1 is not None:
            bool_part[i, j, 0] = case.v1
            bool_mask[i, j, 0] = True
        if case.tu is not None:
            bool_part[i, j, 1] = case.tu
            bool_mask[i, j, 1] = True
        if case.e1 is not None:
            float_part[i, j, 0] = case.e1[1]
            float_mask[i, j, 0] = True
        if case.e4 is not None:
            float_part[i, j, 1] = case.e4[0]
            float_mask[i, j, 1] = True
        if case.f1 is not None:
            float_part[i, j, 2:4] = case.f1
            float_mask[i, j, 2:4] = True
        if case.f2 is not None:
            float_part[i, j, 4:6] = case.f2
            float_mask[i, j, 4:6] = True
        if case.f3 is not None:
            float_part[i, j, 6:8] = case.f3
            float_mask[i, j, 6:8] = True
        if case.f4 is not None:
            float_part[i, j, 8:10] = case.f4
            float_mask[i, j, 8:10] = True
    # 维度顺序转换为 Conv2D 的格式, 并转为 torch.Tensor
    bool_part = torch.tensor(bool_part.transpose((2, 0, 1)), dtype=torch.bool)      # 2 × M × N
    bool_mask = torch.tensor(bool_mask.transpose((2, 0, 1)), dtype=torch.bool)      # 2 × M × N
    float_part = torch.tensor(float_part.transpose((2, 0, 1)), dtype=torch.float32) # 10 × M × N
    float_mask = torch.tensor(float_mask.transpose((2, 0, 1)), dtype=torch.bool)    # 10 × M × N
    return bool_part, bool_mask, float_part, float_mask


# 传入紧凑表示, 返回 case_mat (少一行, 少一列, 填补为 Case=-1)
def recover_case_mat_from_compact(bool_part: torch.Tensor, float_part: torch.Tensor) -> np.ndarray:
    # 注: 除了 padding 的一行一列, 其它均不可能是 Case=-1
    nr_blocks_vert = bool_part.shape[1]
    nr_blocks_horiz = bool_part.shape[2]
    case_mat = np.zeros(shape=(nr_blocks_vert, nr_blocks_horiz), dtype=Case)
    case_mat[:, -1] = Case(no=-1)
    case_mat[-1, :] = Case(no=-1)
    for i, j in product(range(nr_blocks_vert - 1), range(nr_blocks_horiz - 1)):
        # 尽力获取每个信息 (注: None 可能源自 [自己/右/下/右下] 有 invalid 的, 也可能本来就是 None)
        v1 = bool(bool_part[0, i, j])
        v2 = bool(bool_part[0, i, j+1])
        v3 = bool(bool_part[0, i+1, j+1])
        v4 = bool(bool_part[0, i+1, j])
        tu = bool(bool_part[1, i, j])
        e1 = (0, float_part[0, i, j])
        e2 = (float_part[1, i, j+1], 1)
        e3 = (1, float_part[0, i+1, j])
        e4 = (float_part[1, i, j], 0)
        f1 = float_part[2:4, i, j].numpy()
        f2 = float_part[4:6, i, j].numpy()
        f3 = float_part[6:8, i, j].numpy()
        f4 = float_part[8:10, i, j].numpy()
        no = cornerType_2_caseNum.get((v1, v2, v3, v4), None)
        if no in [1415, 1617]:
            no = cornerType_2_caseNum[(v1, v2, v3, v4, tu)]
        if no is None: case_mat[i, j] = Case(no=-1)
        elif no == 0: case_mat[i, j] = Case(no=0, v1=v1)
        elif no == 1: case_mat[i, j] = Case(no=1, v1=v1)
        elif no == 2: case_mat[i, j] = Case(no=2, v1=v1, e4=e4, e1=e1, f1=f1)
        elif no == 3: case_mat[i, j] = Case(no=3, v1=v1, e1=e1, e2=e2, f2=f2)
        elif no == 4: case_mat[i, j] = Case(no=4, v1=v1, e2=e2, e3=e3, f3=f3)
        elif no == 5: case_mat[i, j] = Case(no=5, v1=v1, e3=e3, e4=e4, f4=f4)
        elif no == 6: case_mat[i, j] = Case(no=6, v1=v1, e4=e4, e1=e1, f1=f1)
        elif no == 7: case_mat[i, j] = Case(no=7, v1=v1, e1=e1, e2=e2, f2=f2)
        elif no == 8: case_mat[i, j] = Case(no=8, v1=v1, e2=e2, e3=e3, f3=f3)
        elif no == 9: case_mat[i, j] = Case(no=9, v1=v1, e3=e3, e4=e4, f4=f4)
        elif no == 10: case_mat[i, j] = Case(no=10, v1=v1, e4=e4, e2=e2, f4=f4, f3=f3)
        elif no == 11: case_mat[i, j] = Case(no=11, v1=v1, e4=e4, e2=e2, f1=f1, f2=f2)
        elif no == 12: case_mat[i, j] = Case(no=12, v1=v1, e1=e1, e3=e3, f2=f2, f3=f3)
        elif no == 13: case_mat[i, j] = Case(no=13, v1=v1, e1=e1, e3=e3, f1=f1, f4=f4)
        elif no == 14: case_mat[i, j] = Case(no=14, tu=tu, v1=v1, e1=e1, e2=e2, e3=e3, e4=e4, f1=f1, f3=f3)
        elif no == 15: case_mat[i, j] = Case(no=15, v1=v1, tu=tu, e1=e1, e2=e2, e3=e3, e4=e4, f2=f2, f4=f4)
        elif no == 16: case_mat[i, j] = Case(no=16, v1=v1, tu=tu, e1=e1, e2=e2, e3=e3, e4=e4, f2=f2, f4=f4)
        elif no == 17: case_mat[i, j] = Case(no=17, v1=v1, tu=tu, e1=e1, e2=e2, e3=e3, e4=e4, f1=f1, f3=f3)
        else: case_mat[i, j] = Case(no=-1)
    return case_mat


# 传入 [0, 255] 高清图, 返回和 case_mat 同尺寸的, 但加了噪声和模糊的图片.
def highres_to_lowres_imgs(imH: np.ndarray, target_h: int, target_w: int, *, amount: int = 1, shuffle: bool = False) -> list[np.ndarray]:
    # 记 H = high, M = target, L = target // 2
    HToM = lambda img: cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    LToM = lambda img: cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)    # 区分 HToM 和 LToM 是因为降采样最好用 AREA 方法
    ToL = lambda img: cv2.resize(img, (target_w // 2, target_h // 2), interpolation=cv2.INTER_AREA)
    Bin = lambda img: ((img > binarize_threshold) * 255).astype("u1")
    EPS = 1e-5
    ClipUint8 = lambda img: img.clip(0, 255).astype("u1")
    SetWhite = lambda img, *, mask: np.where(mask, 255, img)
    EdgeMask = lambda img: (abs(laplacian_edge_detector(img)) > EPS)
    RandPepper01Mask = lambda img, *, gain: (np.random.rand(*img.shape) < broken_intensity * gain)
    RandIntMat = lambda img, *, amp: np.random.randint(-int(amp), int(amp + 1), img.shape)
    Blur = lambda img, *, gain: cv2.GaussianBlur(img, (0, 0), (blur_intensity * img.shape[0] / target_h * gain))
    # 生成 amount 轮, 每轮 8 张图
    imgs = []
    for _ in range(amount):
        binarize_threshold = np.clip(127 + (np.random.randn() * (30 / 2)), 0, 255)    # 95% 落入 127 ± 30. (该阈值越大, 白色越少, 黑色笔画越粗)
        broken_intensity = np.clip(0.05 + (np.random.randn() * (0.05 / 2)), 0, 1)     # 95% 落入 0.05 ± 0.05. (边界上的点的被破坏的概率)
        blur_intensity = np.clip(1.0 + (np.random.randn() * (0.5 / 2)), 0.1, 2.0)     # 95% 落入 1.0 ± 0.5.

        # Type 0. ToM(imH): 最直白准确的降采样
        imM = HToM(imH)
        im0 = imM

        # Type 1. ToM(ToL(imH)): 进行 [先降采样, 再上采样] 的摧残
        imL = ToL(imM)
        im1 = LToM(imL)

        # Type 2. ToM(Bin(ToL(imM))): 进行 [先降采样, 再二值化, 再上采样] 的摧残
        im2 = LToM(Bin(imL))

        # Type 3. Bin(imM): 最直白准确的二值化
        imMBin = Bin(imM)
        im3 = imMBin

        # Type 4. Brk(imBin): 对 imMBin [边缘随机破坏]
        edgeMaskM = EdgeMask(imMBin)
        imMBinBroken = SetWhite(imMBin, mask=(edgeMaskM & RandPepper01Mask(imMBin, gain=1.0)))
        im4 = imMBinBroken

        # Type 5. Blur(Brk(imBin)): 对 imMBin [边缘随机破坏] 后 [高斯模糊]
        im5 = Blur(imMBinBroken, gain=1.0)

        # Type 6. ToM(Blur(imH)): 高斯模糊后, 降采样. (这里模糊不是为了更好地降采样, 而是为了生成更为模糊的图)
        im6 = HToM(Blur(imH, gain=1.0))

        # Type 7. ToM(Blur(Disturb(imH))): 随机扰动 (Disturb) 后, 高斯模糊, 然后降采样
        dilatedEdgeMaskH = (Blur(EdgeMask(imH).astype("f4"), gain=0.5) > EPS)
        imHDisturb = ClipUint8(imH + dilatedEdgeMaskH * RandPepper01Mask(imH, gain=4.0) * RandIntMat(imH, amp=255))
        im7 = HToM(Blur(imHDisturb, gain=0.5))

        imgs.extend([im0, im1, im2, im3, im4, im5, im6, im7])

    if shuffle:
        np.random.shuffle(imgs)

    return imgs


# 将一个 svg 文件转为若干个 lowRes 图片 + 相应的 Tensor GT, 输出到指定文件夹
def svg_to_dataset(data_idx: int,
                   svg_filename: Path,
                   output_folder: Path,
                   nr_lrImgs_per_svg: int,
                   nr_blocks_vert: int,
                   nr_blocks_horiz: int,
                   subdiv_per_block: int,
                   debug_visualize_case_mat: bool,
                   debug_verify_tensor: bool,
                   debug_save_lr_pngs: bool,
                   ):

    print(f"正在处理 {data_idx} ({svg_filename.name}) ...")

    # 读入 .svg 文件, 转为 grid
    grid = svg_to_grid(svg_filename, nr_blocks_vert, nr_blocks_horiz, subdiv_per_block)

    # 对 grid 随机偏移, 作为 grid 的数据扩充
    grid = random_shifted_grid(grid, subdiv_per_block)

    # 计算 case_mat
    case_mat = grid_to_case_mat(grid, nr_blocks_vert, nr_blocks_horiz, subdiv_per_block)

    # 将 case_mat 存储为 .svg
    if debug_visualize_case_mat:
        # 注: 此时 case_mat 中无法被确定的会被标记为 no=-1,
        # 但一会转成 Tensor 之后就丢失了这一层意味,
        # 于是从 Tensor 恢复出的 case_mat 就不再包含 no=-1 的情形.
        case_mat_to_svg(case_mat, output_folder / f"NMC_{data_idx}.svg", draw_nodes=True, draw_grids=True)

    # 将 case_mat 转为四张量表示
    bool_part, bool_mask, float_part, float_mask = case_mat_to_compact(case_mat)
    torch.save([bool_part, bool_mask, float_part, float_mask], output_folder / f"Y_{data_idx}.pt")

    # 将四张量表示转回 case_mat, 用于验证正确性
    if debug_verify_tensor:
        # 将 Tensor 转为 case_mat, 然后输出 .svg
        case_mat_recovered = recover_case_mat_from_compact(bool_part, float_part)
        case_mat_to_svg(case_mat_recovered, output_folder / f"NMC_{data_idx}_verify.svg", draw_nodes=True, draw_grids=True)
        # 将 Tensor 随机扰动一下, 然后再转为 case_mat, 最后输出 .svg
        random_flip_mask = torch.rand(size=bool_mask.shape) < 0.05
        random_disturb = torch.rand(size=float_part.shape) * 0.05
        new_bool_part = bool_part ^ random_flip_mask
        new_float_part = (float_part + random_disturb).clamp(0, 1)
        new_case_mat_recovered = recover_case_mat_from_compact(new_bool_part, new_float_part)
        case_mat_to_svg(new_case_mat_recovered, output_folder / f"NMC_{data_idx}_verify_disturbed.svg", draw_nodes=True, draw_grids=True)

    # 将 grid ∈ [-1, 1] 转化为略低清晰度的 high_res ∈ [0, 255], 提高运行效率
    high_res = cv2.resize(((1 - grid) * 127.5).astype("u1"), (nr_blocks_horiz * 4, nr_blocks_vert * 4), interpolation=cv2.INTER_AREA)
    Image.fromarray(high_res).save(output_folder / f"X_{data_idx}__HR.png")

    # 生成和 case_mat 同样大小的 PNG 光栅图, 并随机加入噪声、模糊等作为 data augmentation
    low_res_imgs = highres_to_lowres_imgs(high_res, target_h=nr_blocks_vert, target_w=nr_blocks_horiz, amount=nr_lrImgs_per_svg, shuffle=False)
    if debug_save_lr_pngs:
        for img_idx, img in enumerate(low_res_imgs):
            Image.fromarray(img).save(output_folder / f"X_{data_idx}_{img_idx}.png")

    # 压缩成一个 Tensor 存储, 尺寸为 [len(low_res_imgs), nr_blocks_vert, nr_blocks_horiz]
    imgs_packed = np.concatenate(low_res_imgs).reshape((len(low_res_imgs), nr_blocks_vert, nr_blocks_horiz))
    to_tensor = torch.tensor(imgs_packed, dtype=torch.float32)
    torch.save(to_tensor, output_folder / f"X_{data_idx}.pt")

    print(f"✅  {data_idx} ({svg_filename.name}) 处理完毕!")


if __name__ == "__main__":

    nr_lrImgs_per_svg = 3   # 每个 svg 生成的 lowRes 图片数量 (不同的模糊核, 噪音 etc.)
    nr_blocks_vert = 100
    nr_blocks_horiz = 100
    subdiv_per_block = 64
    svg_folder = Path("./svg/")
    output_folder = Path("./dataset/")

    debug_visualize_case_mat = True
    debug_verify_tensor = True
    debug_save_lr_pngs = True

    svg_files: list[Path] = list(svg_folder.glob("*.svg"))
    svg_files.sort(key=lambda x: str(x))
    random.seed(1028)
    random.shuffle(svg_files)
    # svg_files = svg_files[:8]     # fixme: 随机抽取 k 个 svg 来测试

    print(f"请确认以下参数:\n")
    print(f" * debug_visualize_case_mat = {debug_visualize_case_mat}")
    print(f" * debug_verify_tensor      = {debug_verify_tensor}")
    print(f" * debug_save_lr_pngs       = {debug_save_lr_pngs}")
    print(f" * len(svg_files)           = {len(svg_files)}")
    if input("\n输入 y 以继续: ") != "y":
        print("已取消!")
        exit()

    # TODO: 程序应该输出的是一个 X_highRes_Img 和一个 Y_GT_Tensor,\
    #       而 data 的 [加噪] 要么另起一个程序, 要么在训练过程中动态添加.

    # TODO: 32×32 的数据生成

    if use_mp := True:
        # 分 N_WORKERS 个进程来处理 svg_files
        # N_WORKERS = multiprocessing.cpu_count()
        N_WORKERS = 8
        with multiprocessing.Pool(N_WORKERS) as pool:
            func = functools.partial(
                svg_to_dataset,
                output_folder=output_folder,
                nr_lrImgs_per_svg=nr_lrImgs_per_svg,
                nr_blocks_vert=nr_blocks_vert,
                nr_blocks_horiz=nr_blocks_horiz,
                subdiv_per_block=subdiv_per_block,
                debug_visualize_case_mat=debug_visualize_case_mat,
                debug_verify_tensor=debug_verify_tensor,
                debug_save_lr_pngs=debug_save_lr_pngs,
            )
            pool.starmap(func, enumerate(svg_files))
    else:
        # 单线程, 方便 Debug
        for data_idx, svg_filename in enumerate(svg_files):
            svg_to_dataset(data_idx, svg_filename, output_folder, nr_lrImgs_per_svg, nr_blocks_vert, nr_blocks_horiz, subdiv_per_block)

