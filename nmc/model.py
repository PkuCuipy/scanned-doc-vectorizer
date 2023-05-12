# 2023-05-12

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from make_dataset import recover_case_mat_from_compact, case_mat_to_svg
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResBlock(nn.Module):

    def __init__(self, nr_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=nr_channels, out_channels=nr_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=nr_channels, out_channels=nr_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        output = self.conv1(x)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.conv2(output)
        output = output + x
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output


class CNN(nn.Module):

    def __init__(self, out_bool: bool, out_float: bool, feat_size: int = 128):
        super().__init__()
        self.feat_size = feat_size
        self.out_bool = out_bool
        self.out_float = out_float
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=self.feat_size, kernel_size=3, stride=1, padding=1, bias=True)               # 感受野 -> 3×3
        self.res1 = ResBlock(self.feat_size)
        self.conv1 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=3, stride=1, padding=1, bias=True)  # 感受野 -> 5×5
        self.res2 = ResBlock(self.feat_size)
        self.conv2 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=3, stride=1, padding=1, bias=True)  # 感受野 -> 7×7
        self.res3 = ResBlock(self.feat_size)
        self.conv3 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=3, stride=1, padding=1, bias=True)  # 感受野 -> 9×9
        self.res4 = ResBlock(self.feat_size)
        self.conv4 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=3, stride=1, padding=1, bias=True)  # 感受野 -> 11×11
        self.res5 = ResBlock(self.feat_size)
        self.conv5 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=3, stride=1, padding=1, bias=True)  # 感受野 -> 13×13
        self.res6 = ResBlock(self.feat_size)
        self.conv6 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=3, stride=1, padding=1, bias=True)  # 感受野 -> 15×15
        self.res7 = ResBlock(self.feat_size)
        self.res8 = ResBlock(self.feat_size)
        self.res9 = ResBlock(self.feat_size)
        self.res10 = ResBlock(self.feat_size)
        self.conv7 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv8 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=1, stride=1, padding=0, bias=True)
        if self.out_bool:
            self.conv_out_bool = nn.Conv2d(in_channels=self.feat_size, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv2d(in_channels=self.feat_size, out_channels=10, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x.shape = (batch_size, 1, H, W)
        out = x
        out = self.conv0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res1(out)
        out = self.conv1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res2(out)
        out = self.conv2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res3(out)
        out = self.conv3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res4(out)
        out = self.conv4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res5(out)
        out = self.conv5(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res6(out)
        out = self.conv6(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res7(out)
        out = self.res8(out)
        out = self.res9(out)
        out = self.res10(out)
        out = self.conv7(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv8(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float


class MyDataset(Dataset):
    def __init__(self, create_X_from_HR: bool):
        self.create_X_from_HR = create_X_from_HR
        self.folder = Path("./dataset")
        self.nr_images = len(list(self.folder.glob("X_*__HR.png" if self.create_X_from_HR else "X_*.pt")))

    def __len__(self):
        return self.nr_images

    def __getitem__(self, idx) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.create_X_from_HR:
            # todo: 读入 HR 图片, 生成 X: (batch_size, 1, H, W)
            raise NotImplementedError
        else:
            X_filepath = self.folder / f"X_{idx}.pt"
            X = torch.load(X_filepath).unsqueeze(dim=1).to(DEVICE) * (-1 / 255.0)   # (batch_size, 1, H, W), ∈ [0,1], 0 为外部(背景), 1 为内部(笔画)
        Y_filepath = self.folder / f"Y_{idx}.pt"
        Y_bool_GT, Mask_bool, Y_float_GT, Mask_float = torch.load(Y_filepath)
        Y_bool_GT = torch.unsqueeze(Y_bool_GT, dim=0).to(DEVICE)    # (batch_size, 2, H, W), bool ∈ [True, False]
        Mask_bool = torch.unsqueeze(Mask_bool, dim=0).to(DEVICE)    # (batch_size, 2, H, W), bool ∈ [True, False]
        Y_float_GT = torch.unsqueeze(Y_float_GT, dim=0).to(DEVICE)  # (batch_size, 10, H, W), f32 ∈ [0,1]
        Mask_float = torch.unsqueeze(Mask_float, dim=0).to(DEVICE)  # (batch_size, 10, H, W), bool ∈ [True, False]
        return idx, X, Y_bool_GT, Mask_bool, Y_float_GT, Mask_float



if __name__ == "__main__":

    # 预测的 SVG 输出的保存位置
    output_folder = Path("./output")
    output_folder.mkdir(exist_ok=True)

    # 模型保存位置
    model_save_path = Path("./model_save")
    model_save_path.mkdir(exist_ok=True)

    # 定义模型
    model = CNN(out_bool=True, out_float=True, feat_size=128).to(DEVICE)

    # 加载模型
    load_model_from_file = False
    if load_model_from_file:
        model.load_state_dict(torch.load(model_save_path / "model.pt", map_location=DEVICE))

    # 划分训练集和测试集
    dataset = MyDataset(create_X_from_HR=False)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=None, shuffle=True, num_workers=0)

    # 损失函数
    def loss_func(bool_pred, bool_GT, bool_mask, float_pred, float_GT, float_mask, mix_coef: float):
        def log(x): return torch.log(x + 1e-10)
        loss_bool = (bool_mask * (bool_GT * log(bool_pred) + (~bool_GT) * log(1 - bool_pred))).sum() / (-2 * bool_mask.sum() + 1e-10)
        loss_float = (((float_pred - float_GT) * float_mask).square()).sum() / (10 * float_mask.sum() + 1e-10)
        loss_total = loss_bool + mix_coef * loss_float
        return loss_total, loss_bool, loss_float

    # 训练
    train_model = True
    if train_model:
        NR_EPOCHS = 100
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for epoch in range(NR_EPOCHS):
            for data_idx, X, Y_bool_GT, Mask_bool, Y_float_GT, Mask_float in (pbar := tqdm(train_dataloader, desc=f"epoch {epoch}")):
                Y_bool_pred, Y_float_pred = model(X)
                loss, loss_bool, loss_float = loss_func(Y_bool_pred, Y_bool_GT, Mask_bool, Y_float_pred, Y_float_GT, Mask_float, mix_coef=1.0)
                pbar.set_description_str(f"loss_bool: {loss_bool.item():.6f}, loss_float: {loss_float.item():.6f}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            torch.save(model.state_dict(), model_save_path / f"model_{epoch}.pt")   # save model after each epoch

    # 预测
    all_loss = []
    all_loss_bool = []
    all_loss_float = []
    with torch.no_grad():
        model.eval()
        for data_idx, X, Y_bool_GT, Mask_bool, Y_float_GT, Mask_float in (pbar := tqdm(test_dataloader)):
            Y_bool_pred, Y_float_pred = model(X)
            loss, loss_bool, loss_float = loss_func(Y_bool_pred, Y_bool_GT, Mask_bool, Y_float_pred, Y_float_GT, Mask_float, mix_coef=1.0)
            all_loss.append(loss)
            all_loss_float.append(loss_float)
            all_loss_bool.append(loss_bool)
            for sub_idx in range(Y_bool_pred.shape[0]):
                case_mat_recovered = recover_case_mat_from_compact((Y_bool_pred[sub_idx] > 0.5).cpu(), Y_float_pred[sub_idx].cpu())
                case_mat_to_svg(case_mat_recovered, output_folder / f"predict_{data_idx}_{sub_idx}.svg", draw_nodes=True, draw_grids=True)
            input("按回车键继续...")

    mean_loss = np.mean(all_loss)
    mean_loss_bool = np.mean(all_loss_bool)
    mean_loss_float = np.mean(all_loss_float)
    print(f"test mean_loss: {mean_loss:.6f}")
    print(f"test mean_loss_bool: {mean_loss_bool:.6f}")
    print(f"test mean_loss_float: {mean_loss_float:.6f}")


