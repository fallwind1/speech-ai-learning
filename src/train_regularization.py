import torch
from torch import nn
from torch.utils.data import DataLoader,random_split,Subset
from torchvision import datasets,transforms
import csv
import random
import numpy as np
from pathlib import Path

#CONFIG
CONFIG = {
    "seed":42, "batch_size":64, "hidden_dim":128, "lr":0.1, "epochs":20,
    "data_size":2000, "train_size":1600, "val_size":400
}

#2.设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#3.dataset创建
def build_dataset():
    dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    generator = torch.Generator().manual_seed(CONFIG["seed"])
    subset_indices = torch.randperm(
        len(dataset),
        generator=generator
    )[:CONFIG["data_size"]]
    subset = Subset(
        dataset,
        subset_indices.tolist()
    )

    train_dataset, val_dataset = random_split(
        subset,
        [CONFIG["train_size"],CONFIG["val_size"]],
        generator=generator
    )

    return train_dataset, val_dataset

#4.model
class MLP(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,10)
        )

    def forward(self,x):
        return self.model(x)

#5.训练函数
def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += batch_size * loss.item()
        total_samples += batch_size

        pred = logits.argmax(dim=1)
        correct += (pred==labels).sum().item()

    return total_loss / total_samples, correct / total_samples

#6. 验证函数
def eval_one_epoch(model,loader,criterion,device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += batch_size * loss.item()
            total_samples += batch_size

            pred = logits.argmax(dim=1)
            correct += (pred==labels).sum().item()

    return total_loss / total_samples, correct / total_samples

#7. run_experiment()
def run_experiment(
        experiment_name,
        weight_decay
):
    print(f"Running {experiment_name}")

    set_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = build_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

    model = MLP(CONFIG["hidden_dim"]).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    records = []

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model,train_loader,criterion,optimizer,device
        )

        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        record={
            "epoch":
                epoch+1,
            "experiment":
                experiment_name,
            "train_loss":
                train_loss,
            "val_loss":
                val_loss,
            "train_acc":
                train_acc,
            "val_acc":
                val_acc,
        }

        records.append(record)
        print(record)

    return records

#8. 保存csv
def save_results(results):

    path = Path("results/w02_regularization.csv")

    path.parent.mkdir(
        exist_ok=True
    )

    with open(path,"w",newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "experiment",
                "train_loss",
                "val_loss",
                "train_acc",
                "val_acc"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("saved:",path)

#9. main
def main():
    results = []

    results.extend(run_experiment(
        "no_decay",0
    ))

    results.extend(run_experiment(
        "weight decay",1e-4
    ))

    save_results(results)

if __name__ == "__main__":
    main()