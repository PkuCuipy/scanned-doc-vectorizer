# 2023-03-25

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange

# A CNN model use for Grayscale-Image Super-Resolution
class MySRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x.shape = (batch_size, height, width)
        return self.model(x)


def train_a_batch(model, x, y, optimizer, loss_fn):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def test_single_image(model, img_mat, img_title=""):
    result = model(img_mat).squeeze(0).detach().numpy()
    Image.fromarray(result * 255).show(title=img_title)


if __name__ == "__main__":

    # load model from file
    model = MySRModel()
    try: model.load_state_dict(torch.load("./save/model.pth"))
    except Exception as e: print(e)

    img1 = Image.open("../data/test_2.png").convert("L")
    img1.show()

    img_mat = torch.from_numpy(np.array(img1)).unsqueeze(0).float() / 255.0

    test_single_image(
        model=model,
        img_mat=img_mat,
    )

    for i in trange(300):
        if not i % 10:
            test_single_image(
                model=model,
                img_mat=img_mat,
                img_title=f"Epoch_{i}",
            )
        train_a_batch(
            model=model,
            x=img_mat,
            y=img_mat,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            loss_fn=torch.nn.MSELoss(),
        )

    # save model
    torch.save(model.state_dict(), "./save/model.pth")












