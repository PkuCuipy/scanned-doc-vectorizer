# 2023-05-12

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    def __init__(self, out_bool: bool, out_float: bool):
        super().__init__()
        self.feat_size = 128
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
        self.conv4 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=3, stride=1, padding=0, bias=True)  # 感受野 -> 11×11
        self.res5 = ResBlock(self.feat_size)
        self.conv5 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=3, stride=1, padding=0, bias=True)  # 感受野 -> 13×13
        self.res6 = ResBlock(self.feat_size)
        self.conv6 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=3, stride=1, padding=0, bias=True)  # 感受野 -> 15×15
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

    def forward(self, x):
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


if __name__ == "__main__":

    model = CNN(out_bool=True, out_float=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # # use random data to test (batch_size, channel, height, width)
    # X = torch.randn(8, 1, 100, 100).to(DEVICE)
    # X = F.pad(X, (3, 3, 3, 3), mode="constant", value=X[0, 0, 0, 0])
    # Y_bool_GT = torch.randn(1, 2, 100, 100).to(DEVICE)
    # Y_float_GT = torch.randn(1, 10, 100, 100).to(DEVICE)
    # Mask_bool = (torch.rand(1, 2, 100, 100) < 0.5).to(DEVICE)
    # Mask_float = (torch.rand(1, 10, 100, 100) < 0.5).to(DEVICE)

    # read data from file
    Y_bool_GT, Mask_bool, Y_float_GT, Mask_float = torch.load("dataset/Y_0.pt")
    Y_bool_GT = torch.unsqueeze(Y_bool_GT, dim=0).type(torch.float32).to(DEVICE)
    Mask_bool = torch.unsqueeze(Mask_bool, dim=0).type(torch.float32).to(DEVICE)
    Y_float_GT = torch.unsqueeze(Y_float_GT, dim=0).type(torch.float32).to(DEVICE)
    Mask_float = torch.unsqueeze(Mask_float, dim=0).type(torch.float32).to(DEVICE)
    X = torch.load("dataset/X_0.pt")
    X = torch.unsqueeze(X, dim=1)
    X = F.pad(X, (3, 3, 3, 3), mode="constant", value=X[0, 0, 0, 0]).to(DEVICE)

    for _ in (pbar := tqdm(range(1000))):
        Y_bool_pred, Y_float_pred = model(X)
        loss_bool = ((Y_bool_pred - Y_bool_GT) * Mask_bool).square().mean()
        loss_float = ((Y_float_pred - Y_float_GT) * Mask_float).square().mean()
        loss = loss_bool + loss_float
        pbar.set_description_str(f"loss_bool: {loss_bool.item():.6f}, loss_float: {loss_float.item():.6f}")
        loss.backward()
        optimizer.step()
        optimizer.step()
        optimizer.zero_grad()



