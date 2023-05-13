# 2023-05-12

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image
from tqdm import tqdm
from make_dataset import recover_case_mat_from_compact, case_mat_to_svg
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


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
        if self.out_bool: self.conv_out_bool = nn.Conv2d(in_channels=self.feat_size, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)
        if self.out_float: self.conv_out_float = nn.Conv2d(in_channels=self.feat_size, out_channels=10, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = x                 # x.shape = (batch_size, 1, H, W)
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


# 测试模型在单张图片上给出的 SVG
def test_model_on_single_image(model: torch.nn.Module, greyscale_image_path: Path, save_svg_filepath: Path):
    with torch.no_grad():
        img = Image.open(str(greyscale_image_path)).convert("L")
        tensor = torch.tensor(np.array(img))
        X = tensor.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE) * (-1 / 255.0)  # (1, 1, H, W), ∈ [0,1], 0 为外部(背景), 1 为内部(笔画)
        Y_bool_pred, Y_float_pred = model(X)
        case_mat_recovered = recover_case_mat_from_compact((Y_bool_pred[0] > 0.5).cpu(), Y_float_pred[0].cpu())
        case_mat_to_svg(case_mat_recovered, save_svg_filepath, draw_nodes=True, draw_grids=True)


# 损失函数
def loss_func(bool_pred, bool_GT, bool_mask, float_pred, float_GT, float_mask):
    def log(x): return torch.log(x + 1e-10)
    mix_coef = 1.0
    batch_size = bool_pred.shape[0]
    loss_bool = (bool_mask * (bool_GT * log(bool_pred) + (~bool_GT) * log(1 - bool_pred))).sum() / (-2 * bool_mask.sum() * batch_size + 1e-10)
    loss_float = (((float_pred - float_GT) * float_mask).square()).sum() / (10 * float_mask.sum() * batch_size + 1e-10)
    loss_total = loss_bool + mix_coef * loss_float
    return loss_total, loss_bool, loss_float


# Accuracy 函数
def bool_acc_func(bool_pred, bool_GT, bool_mask) -> float:
    batch_size = bool_pred.shape[0]
    return (bool_mask * ((bool_pred > 0.5) == bool_GT)).sum() / (bool_mask.sum() * batch_size + 1e-10)



if __name__ == "__main__":

    LR = 5e-4
    FEAT = 128
    model_name = f"{datetime.datetime.now().strftime('%Y-%m-%d(%H.%M.%S)')}_lr={LR:.0e}_feat={FEAT}"
    print(model_name)

    # 输出的保存位置
    result_folder = Path("./result") / model_name
    result_folder.mkdir(exist_ok=True)

    # 模型保存位置
    model_save_folder = Path("./model_save") / model_name
    model_save_folder.mkdir(exist_ok=True, parents=True)

    # Tensor Board
    tensorboard_save_folder = Path("./tensorboard_logs") / model_name
    tensorboard_save_folder.mkdir(exist_ok=True, parents=True)
    tb_writer = SummaryWriter(log_dir=str(tensorboard_save_folder))

    # 定义模型
    model = CNN(out_bool=True, out_float=True, feat_size=FEAT).to(DEVICE)

    # 加载模型
    load_epoch = None
    load_model_path = Path("someRandomFolderPath")
    if load_model_path.exists() and (load_epoch is not None):
        model.load_state_dict(torch.load(model_save_folder / f"{load_epoch}.pt", map_location=DEVICE))
        print(f"成功加载模型 {load_model_path.name} 的第 {load_epoch} 个 epoch")
    else:
        print(f"未加载模型, 从头开始训练, 原因: [{load_epoch} is None] 或者 [{load_model_path} 不存在]")

    # 划分训练集和测试集
    dataset = MyDataset(create_X_from_HR=False)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=None, shuffle=True, num_workers=0)

    # 训练
    NR_EPOCHS = 100                     # 总 Epoch 数 (一个 epoch 即完整过了一遍训练集)
    TEST_INTERVAL = 100                 # 每 1000 步进行一次测试集上的测试
    SAVE_MODEL_INTERVAL = 1000          # 每 1000 步保存一次模型
    TEST_IMG2SVG_INTERVAL = 1000        # 每 1000 步测一次图片转 SVG 的输出性能
    LR_SCHEDULER_INTERVAL = 10000       # 每 10000 步降低一次学习率
    LR_SCHEDULER_GAMMA = 0.8            # 每次学习率降低到原来的 0.8 倍
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_SCHEDULER_GAMMA, verbose=True)

    glob_step_cnt = 0

    for epoch in range(NR_EPOCHS):
        for data_idx, X, Y_bool_GT, Mask_bool, Y_float_GT, Mask_float in (pbar := tqdm(train_dataloader, desc=f"epoch_{epoch}")):

            Y_bool_pred, Y_float_pred = model(X)
            loss, loss_bool, loss_float = loss_func(Y_bool_pred, Y_bool_GT, Mask_bool, Y_float_pred, Y_float_GT, Mask_float)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 打印 loss 在屏幕, 并保存在 TensorBoard 中
            pbar.set_description_str(f"loss_bool: {loss_bool.item():.6f}, loss_float: {loss_float.item():.6f}")

            # tensorboard logs
            bool_acc = bool_acc_func(Y_bool_pred, Y_bool_GT, Mask_bool)
            tb_writer.add_scalar(tag="train / loss_float.sqrt()", scalar_value=torch.sqrt(loss_float), global_step=glob_step_cnt)
            tb_writer.add_scalar(tag="train / loss_bool", scalar_value=loss_bool, global_step=glob_step_cnt)
            tb_writer.add_scalar(tag="train / loss", scalar_value=loss, global_step=glob_step_cnt)
            tb_writer.add_scalar(tag="train / 1-bool_Acc", scalar_value=1-bool_acc, global_step=glob_step_cnt)

            # 每隔一段时间, 抽取一个测试集样本进行测试
            if glob_step_cnt % TEST_INTERVAL == 0:
                test_data_idx, test_X, Y_bool_GT, Mask_bool, Y_float_GT, Mask_float = next(iter(test_dataloader))   # 注: 每次 iter() 后会打乱后从头开始, 等价于随机抽样
                Y_bool_pred, Y_float_pred = model(test_X)
                loss, loss_bool, loss_float = loss_func(Y_bool_pred, Y_bool_GT, Mask_bool, Y_float_pred, Y_float_GT, Mask_float)
                bool_acc = bool_acc_func(Y_bool_pred, Y_bool_GT, Mask_bool)
                tb_writer.add_scalar(tag="test / loss_float.sqrt()", scalar_value=torch.sqrt(loss_float), global_step=glob_step_cnt)
                tb_writer.add_scalar(tag="test / loss_bool", scalar_value=loss_bool, global_step=glob_step_cnt)
                tb_writer.add_scalar(tag="test / loss", scalar_value=loss, global_step=glob_step_cnt)
                tb_writer.add_scalar(tag="test / 1-bool_Acc", scalar_value=1-bool_acc, global_step=glob_step_cnt)

            # 每隔一段时间, 保存当前模型参数
            if glob_step_cnt % SAVE_MODEL_INTERVAL == 0:
                torch.save(model.state_dict(), model_save_folder / f"{epoch}_{glob_step_cnt}.pt")

            # 每隔一段时间, 对测试文件夹的所有图片预测 SVG
            if glob_step_cnt % TEST_IMG2SVG_INTERVAL == 0:
                for img_path in Path("./test_images").glob("*.png"):
                    test_model_on_single_image(model, greyscale_image_path=img_path, save_svg_filepath=result_folder / f"{img_path.name}_epoch={epoch}_step={glob_step_cnt}.svg")

            # 每隔一段时间, 降低模型学习率
            if glob_step_cnt % LR_SCHEDULER_INTERVAL == 0:
                scheduler.step()

            glob_step_cnt += 1

        # 每个 epoch 结束后保存模型参数
        torch.save(model.state_dict(), model_save_folder / f"{epoch}.pt")
        scheduler.step()
