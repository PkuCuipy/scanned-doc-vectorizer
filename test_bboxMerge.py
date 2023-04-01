# 2023-03-26(15.44.22)
# read an image of shape H*W, extract connected areas,
# output as an integer matrix, where each area is labeled as a unique integer

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import trange

# read image in greyscale mode, then binarize
grey_img = np.array(cv2.imread("data/test_10.png", cv2.IMREAD_GRAYSCALE))
bin_threshold, bin_img = cv2.threshold(grey_img, 175, 255, cv2.THRESH_BINARY_INV)   # 前景为白色 (255)
plt.imshow(bin_img, cmap="gray")
plt.show()

# connected area extraction
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=4)    # 使用 4-连通, 尽可能缩小每个区域

# merge intersect (use bounding box) labels
def bbox_intersect(x1, y1, w1, h1, x2, y2, w2, h2):
    if x1 + w1 < x2 or x1 > x2 + w2: return False
    if y1 + h1 < y2 or y1 > y2 + h2: return False
    return True

# 统计 (w * h) 和 (area / (w * h)) 的分布
# 如果一个区域的 (w * h) 太大, 且 (area / (w * h)) 太小, 则这个区域不适用于接下来的 merge 操作 (认为是表格线之类的东西), 称为 abnormal 的
stat_of_wh = []
stat_of_area_over_wh = []
for x, y, w, h, area in stats:
    stat_of_wh.append(w * h)
    stat_of_area_over_wh.append(area / (w * h))
abnormal_wh = sorted(stat_of_wh)[-len(stat_of_wh) // 20]
abnormal_area_over_wh = sorted(stat_of_area_over_wh)[len(stat_of_area_over_wh) // 20]
def should_not_merge(w, h, area):
    return w * h > abnormal_wh and area / (w * h) < abnormal_area_over_wh


abnormal_bboxes = {i: (x, y, w, h, area) for i, (x, y, w, h, area) in enumerate(stats) if should_not_merge(w, h, area)}
bboxes = {i: (x, y, w, h, area) for i, (x, y, w, h, area) in enumerate(stats) if not should_not_merge(w, h, area) and i != 0}

# BBox 合并
if DO_MERGE := False:
    for _ in trange(MAX_ITER := 10):
        new_bboxes = dict()
        flag_updated = False
        for (i, (x, y, w, h, area)) in bboxes.items():
            # 检查新增的第 i 个区域是否与之前的某个区域相交, 如果相交, 则需要统计出来, 用于稍后的合并
            to_be_merged = []
            for j, (xj, yj, wj, hj, areaj) in new_bboxes.items():
                if bbox_intersect(x, y, w, h, xj, yj, wj, hj):
                    to_be_merged.append(j)
            # 新增第 i 个区域
            if len(to_be_merged) == 0:              # 没有相交的区域, {i} 是新的区域
                new_bboxes[i] = (x, y, w, h, area)
            else:                                   # 有相交的区域, {i, ...j} 合并成一个新的区域
                flag_updated = True
                print(f"{i} 与 {to_be_merged} 合并")
                xmin, ymin, xmax, ymax, area_total = x, y, x + w, y + h, area
                for j in to_be_merged:
                    xj, yj, wj, hj, areaj = new_bboxes[j]
                    labels[yj:yj+hj+1, xj:xj+wj+1][labels[yj:yj+hj+1, xj:xj+wj+1] == j] = i                         # 更新 label_map
                    xmin, xmax, ymin, ymax = min(xmin, xj), max(xmax, xj + wj), min(ymin, yj), max(ymax, yj + hj)   # 更新 Bbox
                    area_total += areaj                                                                             # 更新面积
                    new_bboxes.pop(j)                                                                               # 删除被合并的区域
                new_bboxes[i] = (xmin, ymin, xmax - xmin, ymax - ymin, area_total)
        bboxes = new_bboxes
        if not flag_updated:
            break

# 最后把 abnormal 的区域也加进来
bboxes.update(abnormal_bboxes)

# show labels
plt.imshow(labels, cmap="jet")

# draw bounding boxes
for i, (x, y, w, h, area) in bboxes.items():
    plt.gca().add_patch(plt.Rectangle((x-0.5, y-0.5), w, h, fill=False, edgecolor="r", linewidth=1))
plt.show()

