import torch
from torch import nn
from pathlib import Path
from torchvision import datasets,transforms
from torch.utils.data import Subset,DataLoader,random_split
import matplotlib.pyplot as plt

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

def main():
    #设置参数
    SEED = 42
    BATCH_SIZE = 64
    TRAIN_SIZE = 1600
    VAL_SIZE = 400
    SUBSET_SIZE = 2000
    EPOCHS = 5
    DATA_DIR = Path("data")
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

    subset_generator = torch.Generator().manual_seed(SEED)
    subset_incides = torch.randperm(
        len(mnist_train),
        generator=subset_generator
    )[:SUBSET_SIZE]
    small_mnist = Subset(
        mnist_train,
        subset_incides.tolist()
    )

    split_generator = torch.Generator().manual_seed(SEED)
    train_dataset,val_dataset = random_split(
        small_mnist,
        [TRAIN_SIZE,VAL_SIZE],
        generator=split_generator
    )

    #创建DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    #设置模型、优化器、loss函数
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = 0.1
    )

    #训练验证循环
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model,train_dataloader,criterion,optimizer)
        val_loss = eval_one_epoch(model,val_dataloader,criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch:{epoch+1}",
            f"train={train_loss:.4f}",
            f"val={val_loss:.4f}"
        )

    fig, axes = plt.subplots(1,2,figsize=(10,4), layout="constrained")
    axes[0].plot(range(EPOCHS),train_losses)
    axes[0].set(title="Training curve", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(range(EPOCHS),val_losses)
    axes[1].set(title="evaluate curve", xlabel="Epoch", ylabel="Loss")
    plt.show()

if __name__ == "__main__":
    main()