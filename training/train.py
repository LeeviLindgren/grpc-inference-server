import argparse
from safetensors.torch import save_file

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.linear_1 = nn.Linear(64 * 7 * 7, 128)
        self.linear_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2d_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.flatten(1)
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        x = F.softmax(x, dim=-1)
        return x


class LitMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvNet()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save the model weights (.safetensors)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Optional path to store/download MNIST data"
    )
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_dir = args.data_dir or "./data"
    mnist_train = MNIST(root=data_dir, train=True, download=True, transform=transform)
    mnist_test = MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=128)

    model = LitMNIST()
    trainer = pl.Trainer(max_epochs=3, log_every_n_steps=10)
    trainer.fit(model, train_loader)

    # Evaluate accuracy on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            y_hat = model(x)
            preds = y_hat.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"\n✅ Final test accuracy: {acc:.4f}")

    # Save weights in SafeTensors format
    state_dict = model.model.state_dict()
    save_file(state_dict, args.output)
    print(f"📦 Saved model weights to: {args.output}")


if __name__ == "__main__":
    main()
