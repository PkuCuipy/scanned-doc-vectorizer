# 2023-05-03
# 传入一个 SVG, 尺寸向上取整为 M × N,
# 那么就对应于 M × N 个小方块, 每个小方块里对应一些边.

# from __future__ import annotations
import fitz
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import cv2
from itertools import product

N_BLOCKS_MAX = 32                               # 网格边上的 [大方块] 数
SUBDIV_PER_BLOCK = 32                            # 每个 [大方块] 边的 [小方块] 细分数 (最小为 1 即不细分)
NR_SUB_BLOCKS = N_BLOCKS_MAX * SUBDIV_PER_BLOCK # 小方块总数 (格点数 = N_SUB_BLOCKS + 1)

# 读入 svg, 然后转 pdf (否则无法编辑 media_box)
svg_doc = fitz.Document("font.svg")
pdf_bytes = svg_doc.convert_to_pdf()
pdf_doc = fitz.Document("pdf", pdf_bytes)
page = pdf_doc.load_page(0)

width = page.rect.width
height = page.rect.height
size = max(width, height)
scale = NR_SUB_BLOCKS / size

# SVG 渲染器得到的是像素值. 为了得到格点值, 将四周扩大 0.5 像素, 这样渲染出的 [像素值] 可以视为原始的 [格点值]
correction = 0.5 / scale
page.set_mediabox(fitz.Rect(-correction, -correction, width+correction, height+correction))
pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace="gray")
img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
mat = np.array(img)
mat = mat[:NR_SUB_BLOCKS + 1, :NR_SUB_BLOCKS + 1]   # 防止浮点误差导致多出来一行/列
grid = -(mat - 127.5) / 127.5                       # 使得 grid ∈ [-1, 1], 且 [+] 为内部, [-] 为外部

# 确保 grid 的尺寸是 N_SUB_DIV 的整倍数 + 1
if grid.shape[0] > grid.shape[1]:   # |
    if (grid.shape[1] - 1) % SUBDIV_PER_BLOCK != 0:
        target_size = ((grid.shape[1] - 1) // SUBDIV_PER_BLOCK + 1) * SUBDIV_PER_BLOCK + 1
        grid = np.hstack([grid, -np.ones((grid.shape[0], target_size - grid.shape[1]))])
else:                               # ——
    if (grid.shape[0] - 1) % SUBDIV_PER_BLOCK != 0:
        target_size = ((grid.shape[0] - 1) // SUBDIV_PER_BLOCK + 1) * SUBDIV_PER_BLOCK + 1
        grid = np.vstack([grid, -np.ones((target_size - grid.shape[0], grid.shape[1]))])
# plt.matshow(grid, cmap="bwr")

nr_block_horiz = grid.shape[1] // SUBDIV_PER_BLOCK
nr_block_vert = grid.shape[0] // SUBDIV_PER_BLOCK

case_mat = np.zeros((nr_block_vert, nr_block_horiz), dtype=np.int32)
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


# 18 Cases of Marching Square
class Case:
    __slots__ = ("no", "v1", "tu", "e1", "e4", "f1", "f2", "f3", "f4")
    def __init__(self, *, no: int   = None,
                          v1: bool  = None, tu: bool  = None,
                          e1: float = None, e4: float = None,
                          f1: tuple[float, float] = None, f2: tuple[float, float] = None,
                          f3: tuple[float, float] = None, f4: tuple[float, float] = None):
        self.no, self.v1, self.tu, self.e1, self.e4, self.f1, self.f2, self.f3, self.f4 = no, v1, tu, e1, e4, f1, f2, f3, f4
    def __repr__(self):
        return f"Case(no={self.no}, v1={self.v1}, tu={self.tu}, \n\t e1={self.e1}, e4={self.e4}, \n\t f1={self.f1}, \n\t f2={self.f2}, \n\t f3={self.f3}, \n\t f4={self.f4})"

# 注: 相邻的共享节点不重复存储. 代价是最后一行和最后一列被舍弃(?)
float_part = np.zeros(shape=(nr_block_vert, nr_block_horiz, 10), dtype=np.float32)  # M × N × 10
float_mask = np.zeros(shape=(nr_block_vert, nr_block_horiz, 10), dtype=bool)        # M × N × 10
bool_part = np.zeros(shape=(nr_block_vert, nr_block_horiz, 2), dtype=bool)          # M × N × 2
bool_mask = np.zeros(shape=(nr_block_vert, nr_block_horiz, 2), dtype=bool)          # M × N × 2
case_mat = [[Case() for j in range(nr_block_horiz)] for i in range(nr_block_vert)]  # M × N


# 计算二维向量的长度的平方. 参数 x 是 (N,2) 的, 返回值是 (N,) 的.
def length_square(x: torch.Tensor) -> torch.Tensor:
    x = x.view(-1, 2)
    return (x ** 2 + 0.001).sum(dim=1)   # 加 0.001 也防止梯度 NaN

# 给定一维数组 edge, 计算 +/- 交界处的位置 e ∈ [0, 1]
def calc_e(edge: np.ndarray) -> float:
    div_left_idx = np.argwhere((edge[0] * edge) > 0).ravel()[-1]    # 找到最后一个和 edge[0] 同号的元素, 即为分界点
    div_right_idx = div_left_idx + 1
    div_left = edge[div_left_idx]
    div_right = edge[div_right_idx]
    left_weight = -div_right / (div_left - div_right)
    right_weight = div_left / (div_left - div_right)
    e = (left_weight * div_left_idx + right_weight * div_right_idx) / (edge.shape[0] - 1)
    return e

# 计算二维平面内 [点集 C] 到 [线段 AB] 的距离
def dist_pts_to_seg(C: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
    AB = B - A
    AC = C - A
    lenAB2 = length_square(AB)
    p = ((AC @ AB) / lenAB2).view(-1, 1).clamp(0.001, 0.999).detach()  # clamp 不用 (0, 1) 是防止梯度 NaN
    AH = p * AB
    CH = AH - AC
    lenCH = torch.sqrt(length_square(CH))   # 加 0.001 也是防止梯度 NaN
    return lenCH

# 优化 [点 p], 使得 [折线段 A——p——B] 拟合 [点集 points]
def optimized_single_p(*, p_init: tuple[float, float], A: tuple[float, float], B: tuple[float, float], points: torch.Tensor) -> tuple[float, float]:
    A = torch.tensor(A, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)
    p = torch.tensor(p_init, dtype=torch.float32).requires_grad_(True)
    optimizer = torch.optim.Adam([p], lr=0.05)
    last_loss = 0
    for i in range(100):
        optimizer.zero_grad()
        loss = torch.mean(torch.minimum(
            dist_pts_to_seg(points, A, p),
            dist_pts_to_seg(points, B, p)
        )) + 0.05 * length_square(A - p) + 0.05 * length_square(p - B)
        loss.backward()
        optimizer.step()
        # print(loss)
        if abs((last_loss - loss) / loss) < 1e-4:
            break
        last_loss = loss
    return tuple(p.detach().clamp(0.001, 0.999).tolist())

# 优化 [点 p1 和 p2], 使得 [折线段 A——p1——p2——B] 拟合 [点集 points]
def optimize_p1_and_p2(*, p1_init: tuple[float, float], p2_init: tuple[float, float], A: tuple[float, float], B: tuple[float, float], points: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float]]:
    A = torch.tensor(A)
    B = torch.tensor(B)
    p1 = torch.tensor(p1_init).requires_grad_(True)
    p2 = torch.tensor(p2_init).requires_grad_(True)
    optimizer = torch.optim.Adam([p1, p2], lr=0.05)
    last_loss = 0
    for i in range(100):
        optimizer.zero_grad()
        loss = torch.mean(torch.minimum(torch.minimum(
            dist_pts_to_seg(points, A, p1),
            dist_pts_to_seg(points, p1, p2)),
            dist_pts_to_seg(points, p2, B))) + 0.033 * length_square(A - p1) + 0.033 * length_square(p1 - p2) + 0.033 * length_square(p2 - B)
        loss.backward()
        optimizer.step()
        # print(loss)
        if abs((last_loss - loss) / loss) < 1e-4:
            break
        last_loss = loss
    return tuple(p1.detach().clamp(0.001, 0.999).tolist()), tuple(p2.detach().clamp(0.001, 0.999).tolist())

# 返回 label_mat 中 label 表征的区域对应的边界点的坐标 ∈ [0,1]×[0,1]
def edge_points(label_mat: np.ndarray, label: int) -> torch.Tensor:
    area = (label_mat == label).astype("u1")
    edge = cv2.Laplacian(area, -1)  # fixme: 这个不太准
    ijs = np.argwhere(edge)
    xys = torch.tensor(ijs) / (label_mat.shape[0] - 1)  # xy ∈ [0,1]×[0,1]  # fixme: 这里也要相应地修改
    return xys


# todo: 拆成 3 个循环, 第一个计算 v1, 第二个计算 e1, e4, 第三个计算 f1, f2, f3, f4
# todo: 别在乎效率, 重复计算就重复计算, 先算出来再说, 再者这理论上是离线处理, (大头应该是nn训练,) 时间长就长呗!!!!!
for i, j in product(range(nr_block_vert), range(nr_block_horiz)):
    # 取出当前对应的子矩阵 block
    block = grid[i*SUBDIV_PER_BLOCK:(i+1)*SUBDIV_PER_BLOCK+1, j*SUBDIV_PER_BLOCK:(j+1)*SUBDIV_PER_BLOCK+1]
    top_edge = block[0, :]
    left_edge = block[:, 0]
    right_edge = block[:, -1]
    bottom_edge = block[-1, :]

    # 利用 [四角] 的值初步判断 case number
    corner_type = (block[0, 0] > 0, block[0, -1] > 0, block[-1, -1] > 0, block[-1, 0] > 0)
    case_num = cornerType_2_caseNum[corner_type]

    # 利用 [连通域] 判别 case 的合法性, 以及区分 14 ↔ 15 和 16 ↔ 17
    # 注: 只用 pos_island 是不够的, 因为不能排除其内部中空的情形. 比如一个全 1 方阵, 中心有个 0, 这个是非法 case, 但如果只检查 positive 下的连通性, 则会被错误地归为 case 1.
    nr_labels_pos, label_islands_pos = cv2.connectedComponents((positive:=(block>0).astype("u1")), connectivity=4)    # nr_labels_pos := [正岛屿]个数 + 1(背景)
    nr_labels_neg, label_islands_neg = cv2.connectedComponents(1 - positive, connectivity=4)                          # nr_labels_neg := [负岛屿]个数 + 1(背景)
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
    case = case_mat[i][j]
    if case_num == 0:
        case.v1 = False
    elif case_num == 1:
        case.v1 = True
    elif case_num == 2:
        case.v1 = True
        case.e1 = calc_e(top_edge)
        case.e4 = calc_e(left_edge)
        case.f1 = optimized_single_p(p_init=(0.5, 0.5), A=(0, case.e1), B=(case.e4, 0), points=edge_points(label_islands_pos, label_islands_pos[0, 0]))
    elif case_num == 3:
        case.v1 = False
    elif case_num == 4:
        case.v1 = False
    elif case_num == 5:
        case.v1 = False
    elif case_num == 6:
        case.v1 = False
    elif case_num == 7:
        case.v1 = True
    elif case_num == 8:
        case.v1 = True
    elif case_num == 9:
        case.v1 = True
    elif case_num == 10:
        case.v1 = True
    elif case_num == 11:
        case.v1 = False
    elif case_num == 12:
        case.v1 = True
    elif case_num == 13:
        case.v1 = False
    elif case_num == 14:
        case.v1 = True
    elif case_num == 15:
        case.v1 = True
    elif case_num == 16:
        case.v1 = False
    elif case_num == 17:
        case.v1 = False














# plt.matshow(case_mat, cmap="rainbow")






































