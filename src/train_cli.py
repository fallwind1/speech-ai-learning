import torch
from torch import nn
from pathlib import Path
from torchvision import datasets,transforms
from torch.utils.data import Subset,DataLoader,random_split
import matplotlib.pyplot as plt
import argparse

print("train_cli.py started")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        return self.model(x)

#构建单epoch训练循环
def train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer
):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits,labels)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected.")

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples

    return average_loss

def eval_one_epoch(
        model,
        dataloader,
        criterion
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images)
            loss = criterion(logits,labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    average_loss = total_loss / total_samples

    return average_loss

#读取命令行参数
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr",type=float,default=0.1)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--epochs",type=int,default=20)

    return parser.parse_args()

def main():
    #设置参数以及绝对路径
    args = parse_args()

    CONFIG = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": 42,
        "train_size": 1600,
        "subset_size": 2000,
        "val_size": 400
    }

    print("config:",CONFIG)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    train_losses = []
    val_losses = []

    #设置数据集
    transform = transforms.ToTensor()

    mnist_train = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        transform=transform,
        download=True
    )

    subset_generator = torch.Generator().manual_seed(CONFIG["seed"])
    subset_incides = torch.randperm(
        len(mnist_train),
        generator=subset_generator
    )[:CONFIG["subset_size"]]
    small_mnist = Subset(
        mnist_train,
        subset_incides.tolist()
    )

    split_generator = torch.Generator().manual_seed(CONFIG["seed"])
    train_dataset,val_dataset = random_split(
        small_mnist,
        [CONFIG["train_size"],CONFIG["val_size"]],
        generator=split_generator
    )

    #创建DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

    #设置模型、优化器、loss函数
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = CONFIG["lr"]
    )

    #训练验证循环
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_one_epoch(model,train_dataloader,criterion,optimizer)
        val_loss = eval_one_epoch(model,val_dataloader,criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch:{epoch+1}",
            f"train={train_loss:.4f}",
            f"val={val_loss:.4f}"
        )

    #保存模型
    checkpoint_path = CHECKPOINT_DIR / "w2d6_mnist.pt"
    torch.save(
        {"model_state_dict":model.state_dict(),
        "config":CONFIG},
        checkpoint_path
    )

    fig, axes = plt.subplots(1,2,figsize=(10,4), layout="constrained")
    axes[0].plot(range(1,CONFIG["epochs"]+1),train_losses)
    axes[0].set(title="Training curve", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(range(1,CONFIG["epochs"]+1),val_losses)
    axes[1].set(title="evaluate curve", xlabel="Epoch", ylabel="Loss")
    plt.show()

if __name__ == "__main__":
    main()