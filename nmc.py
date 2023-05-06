# 2023-05-03
# 传入一个 SVG, 尺寸向上取整为 M × N,
# 那么就对应于 M × N 个小方块, 每个小方块里对应一些边.

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

# 传入一个 .svg 文件, 进行网格密集采样  (注: 其实我们完全不 care SVG 的矢量表示, 只是因为这玩意可以超高精度采样, 比如如果有 8K 的字符光栅图像, 那理论上也是 ok 的)
def svg_to_grid(svg_file_path: str, n_blocks_vert: int, n_blocks_horiz: int, n_subdiv: int) -> np.ndarray:
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
    svg_doc = fitz.Document("svg", svg_bytes)
    page = svg_doc.load_page(0)
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
    # 将 mat[][] 放置在 grid[][] 的中心
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
def shifted_grid(grid:np.ndarray, shift_x: float, shift_y: float, block_size: int):
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
    EARLY_BRK_THR = 1e-4    # 基于 loss 的优化早停阈值
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

# 返回 label_mat 中 label 表征的区域对应的边界点的坐标 ∈ [0,1]×[0,1]
def edge_points(label_mat: np.ndarray, label: int, *, laplacian_kernel=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])) -> np.ndarray:
    area = (label_mat == label)
    edge = convolve2d(area, laplacian_kernel, boundary="symm", mode="same")
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
    nr_labels_pos, pos_islands = cv2.connectedComponents((positive:=(block_grid > 0).astype("u1")), connectivity=4)  # nr_labels_pos := [正岛屿]个数 + 1(背景)
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

# 传入整个 grid, 返回和 case_mat 同尺寸的光栅化图片. (这里会对高清图片进行模糊等处理后再降采样)
def grid_to_img(grid: np.ndarray, img_h: int, img_w: int) -> Image.Image:
    high_res = ((1 - grid) * 127.5).astype("u1")
    high_res = cv2.resize(high_res, (img_h * 4, img_w * 4), interpolation=cv2.INTER_LINEAR)   # 没必要那么高分辨率, 提高运行效率
    sigma = (high_res.shape[0] / img_h) * 1.0
    ker_size = int(sigma * 6) // 2 + 1
    blurred = cv2.GaussianBlur(high_res, (ker_size, ker_size), sigma)
    low_res = cv2.resize(blurred, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(low_res)

# 传入 case_mat, 导出到 .svg 文件
def case_mat_to_svg(case_mat: np.ndarray, svg_save_path: str, *, draw_nodes=True, draw_grids=True) -> None:
    dwg = svgwrite.Drawing(filename=svg_save_path, size=("100%", "100%"), viewBox=("0 0 %d %d" % (case_mat.shape[1] + 1, case_mat.shape[0] + 1)))
    polylines: list[list[np.ndarray]] = []
    circles: list[np.ndarray] = []
    for i, j in tqdm(product(range(case_mat.shape[0]), range(case_mat.shape[1])), total=case_mat.size, desc="case_mat -> svg"):
        c = case_mat[i][j]
        if c.no == 2: polylines.append([c.e4 + [i, j], c.f1 + [i, j], c.e1 + [i, j]]); circles.append(c.f1 + [i, j])
        elif c.no == 3: polylines.append([c.e1 + [i, j], c.f2 + [i, j], c.e2 + [i, j]]); circles.append(c.f2 + [i, j])
        elif c.no == 4: polylines.append([c.e2 + [i, j], c.f3 + [i, j], c.e3 + [i, j]]); circles.append(c.f3 + [i, j])
        elif c.no == 5: polylines.append([c.e3 + [i, j], c.f4 + [i, j], c.e4 + [i, j]]); circles.append(c.f4 + [i, j])
        elif c.no == 6: polylines.append([c.e4 + [i, j], c.f1 + [i, j], c.e1 + [i, j]]); circles.append(c.f1 + [i, j])
        elif c.no == 7: polylines.append([c.e1 + [i, j], c.f2 + [i, j], c.e2 + [i, j]]); circles.append(c.f2 + [i, j])
        elif c.no == 8: polylines.append([c.e2 + [i, j], c.f3 + [i, j], c.e3 + [i, j]]); circles.append(c.f3 + [i, j])
        elif c.no == 9: polylines.append([c.e3 + [i, j], c.f4 + [i, j], c.e4 + [i, j]]); circles.append(c.f4 + [i, j])
        elif c.no == 10: polylines.append([c.e4 + [i, j], c.f4 + [i, j], c.f3 + [i, j], c.e2 + [i, j]]); circles.extend([c.f4 + [i, j], c.f3 + [i, j]])
        elif c.no == 11: polylines.append([c.e4 + [i, j], c.f1 + [i, j], c.f2 + [i, j], c.e2 + [i, j]]); circles.extend([c.f1 + [i, j], c.f2 + [i, j]])
        elif c.no == 12: polylines.append([c.e1 + [i, j], c.f2 + [i, j], c.f3 + [i, j], c.e3 + [i, j]]); circles.extend([c.f2 + [i, j], c.f3 + [i, j]])
        elif c.no == 13: polylines.append([c.e1 + [i, j], c.f1 + [i, j], c.f4 + [i, j], c.e3 + [i, j]]); circles.extend([c.f1 + [i, j], c.f4 + [i, j]])
        elif c.no == 14: polylines.append([c.e4 + [i, j], c.f1 + [i, j], c.e1 + [i, j]]); polylines.append([c.e3 + [i, j], c.f3 + [i, j], c.e2 + [i, j]]); circles.extend([c.f1 + [i, j], c.f3 + [i, j]])
        elif c.no == 15: polylines.append([c.e4 + [i, j], c.f4 + [i, j], c.e3 + [i, j]]); polylines.append([c.e1 + [i, j], c.f2 + [i, j], c.e2 + [i, j]]); circles.extend([c.f4 + [i, j], c.f2 + [i, j]])
        elif c.no == 16: polylines.append([c.e4 + [i, j], c.f4 + [i, j], c.e3 + [i, j]]); polylines.append([c.e1 + [i, j], c.f2 + [i, j], c.e2 + [i, j]]); circles.extend([c.f4 + [i, j], c.f2 + [i, j]])
        elif c.no == 17: polylines.append([c.e4 + [i, j], c.f1 + [i, j], c.e1 + [i, j]]); polylines.append([c.e3 + [i, j], c.f3 + [i, j], c.e2 + [i, j]]); circles.extend([c.f1 + [i, j], c.f3 + [i, j]])
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
    float_mask = np.zeros(shape=(*case_mat.shape, 10), dtype=bool)  # M × N × 10
    for i, j in tqdm(product(range(nr_blocks_vert), range(nr_blocks_horiz)), total=case_mat.size, desc="case_mat -> 4 Tensors"):
        case = case_mat[i][j]
        if case.v1 is not None:
            bool_part[i][j][0] = case.v1
            bool_mask[i][j][0] = True
        if case.tu is not None:
            bool_part[i][j][1] = case.tu
            bool_mask[i][j][1] = True
        if case.e1 is not None:
            float_part[i][j][0] = case.e1[1]
            float_mask[i][j][0] = True
        if case.e4 is not None:
            float_part[i][j][1] = case.e4[0]
            float_mask[i][j][1] = True
        if case.f1 is not None:
            float_part[i][j][2:4] = case.f1
            float_mask[i][j][2:4] = True
        if case.f2 is not None:
            float_part[i][j][4:6] = case.f2
            float_mask[i][j][4:6] = True
        if case.f3 is not None:
            float_part[i][j][6:8] = case.f3
            float_mask[i][j][6:8] = True
        if case.f4 is not None:
            float_part[i][j][8:10] = case.f4
            float_mask[i][j][8:10] = True
    # 维度顺序转换为 Conv2D 的格式, 并转为 torch.Tensor
    bool_part = torch.tensor(bool_part.transpose((2, 0, 1)), dtype=torch.bool)      # 2 × M × N
    bool_mask = torch.tensor(bool_mask.transpose((2, 0, 1)), dtype=torch.bool)      # 2 × M × N
    float_part = torch.tensor(float_part.transpose((2, 0, 1)), dtype=torch.float32) # 10 × M × N
    float_mask = torch.tensor(float_mask.transpose((2, 0, 1)), dtype=torch.bool)    # 10 × M × N
    return bool_part, bool_mask, float_part, float_mask



if __name__ == "__main__":

    # 读入 .svg 文件, 转为 grid
    nr_blocks_vert = 64
    nr_blocks_horiz = 64
    subdiv_per_block = 64
    grid = svg_to_grid("font.svg", nr_blocks_vert, nr_blocks_horiz, subdiv_per_block)

    # 随机生成若干个 data pair
    for data_idx in range(10):

        # 对 grid 随机偏移, 作为 data augmentation
        dx = np.random.uniform(-1, 1)
        dy = np.random.uniform(-1, 1)
        grid = shifted_grid(grid, dx, dy, subdiv_per_block)

        # 对每个 block 计算相应的 Case 的各种参数
        case_mat = np.zeros(shape=(nr_blocks_vert, nr_blocks_horiz), dtype=Case)
        for i, j in tqdm(product(range(case_mat.shape[0]), range(case_mat.shape[1])), total=case_mat.size, desc="grid -> case_mat"):
            block_grid = grid[i * subdiv_per_block : (i + 1) * subdiv_per_block + 1, j * subdiv_per_block:(j + 1) * subdiv_per_block + 1]  # 取出当前子矩阵对应的 sub_grid
            case_mat[i][j] = block_to_case(block_grid)

        # 将 case_mat 存储为 .svg
        case_mat_to_svg(case_mat, f"./result/nmc_output_{data_idx}.svg", draw_nodes=True, draw_grids=True)

        # 将 case_mat 转为四张量表示
        bool_part, bool_mask, float_part, float_mask = case_mat_to_compact(case_mat)
        torch.save([bool_part, bool_mask, float_part, float_mask], f"./result/nmc_output_{data_idx}.pt")

        # 将 grid 存储为和 case_mat 同样大小的 .png 光栅图
        im = grid_to_img(grid, img_h=nr_blocks_vert, img_w=nr_blocks_horiz)
        im.save(f"./result/nmc_output_{data_idx}.png")

