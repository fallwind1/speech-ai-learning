import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset,random_split
from pathlib import Path
import csv
import matplotlib.pyplot as plt

#1.配置参数
CONFIG = {
    "seed":42,
    "batch_size":64,
    "epochs":20,
    "subset_size":2000,
    "train_size":1600,
    "val_size":400,
    "lr":0.01,
    "hidden_dim":128
}

#2.seed
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

#3.MNIST小数据集和train/val划分
def build_dataset(DATA_DIR,config):
    mnist_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    subset_generator = torch.Generator().manual_seed(config["seed"])
    subset_indices = torch.randperm(
        len(mnist_dataset),
        generator=subset_generator
    )[:config["subset_size"]]
    subset_dataset = Subset(
        mnist_dataset,
        subset_indices.tolist()
    )

    split_generator = torch.Generator().manual_seed(config["seed"])
    train_dataset, val_dataset = random_split(
        subset_dataset,
        [config["train_size"],config["val_size"]],
        generator=split_generator
    )

    return train_dataset, val_dataset

def make_train_loader(train_dataset,config):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    return train_loader

def make_eval_loader(val_dataset,config):
    return DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

#5.MLP
class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,10)
        )

    def forward(self,x):
        return self.model(x)

#6.train_one_epoch
def train_one_epoch(model,dataloader,criterion,optimizer,device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images,labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += batch_size * loss.item()
        total_samples += batch_size

    return total_loss / total_samples

#7.eval_one_epoch
def eval_one_epoch(model,dataloader,criterion,device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits,labels)

            batch_size = images.size(0)
            total_loss += batch_size * loss.item()
            total_samples += batch_size

    return total_loss / total_samples

#save csv
def save_csv(RESULT_DIR,sample_loss):
    csv_path = RESULT_DIR / "w02_satge_check.csv"

    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss"
            ]
        )
        writer.writeheader()

        for sample in sample_loss:
            writer.writerow(
                {
                    "epoch":sample["epoch"],
                    "train_loss":sample["train_loss"],
                    "val_loss":sample["val_loss"]
                }
            )


#8. main
def main():
    #文件路径
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_DIR / "data"
    FIGURE_DIR = PROJECT_DIR / "figures"
    RESULT_DIR = PROJECT_DIR / "results"
    FIGURE_DIR.mkdir(parents=True,exist_ok=True)
    RESULT_DIR.mkdir(parents=True,exist_ok=True)

    #实验配置
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(CONFIG["hidden_dim"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG["lr"]
    )

    train_dataset, val_dataset = build_dataset(
        DATA_DIR,CONFIG
    )
    train_loader = make_train_loader(train_dataset,CONFIG)
    val_loader = make_eval_loader(val_dataset,CONFIG)

    sample_loss = []
    train_losses = []
    val_losses = []

    #训练验证循环
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_one_epoch(model,train_loader,criterion,optimizer,device)
        val_loss = eval_one_epoch(model,val_loader,criterion,device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"{epoch+1}轮的train_loss:{train_loss:.4f}\tval_loss:{val_loss:.4f}")

        sample_loss.append(
            {
                "epoch":epoch,
                "train_loss":train_loss,
                "val_loss":val_loss
            }
        )

    #保存每轮loss为csv文件
    save_csv(RESULT_DIR,sample_loss)

    #绘制loss曲线
    fig, ax = plt.subplots(figsize=(12,6),layout="constrained")
    ax.plot(range(1,CONFIG["epochs"]+1),train_losses,label="train")
    ax.plot(range(1,CONFIG["epochs"]+1),val_losses,label="val")
    ax.set(title="loss",xlabel="epoch", ylabel="loss")
    ax.legend()
    fig.savefig(
        FIGURE_DIR / "w02_stage_check.png",
        dpi = 150
    )
    plt.show()

if __name__ == "__main__":
    main()