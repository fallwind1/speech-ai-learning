"""
模型预测
↓
找到 preds != labels
↓
收集前5个错分样本
↓
记录 true / pred / confidence
↓
保存CSV
"""
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Subset,
    random_split,
)
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
# =========================================================
# 1. 项目路径
# =========================================================
PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_PATH / "data"
CHECKPOINT_DIR = PROJECT_PATH / "checkpoints" / "w2d6_mnist.pt"
RESULTS_DIR = PROJECT_PATH / "results"
FIGURES_DIR = PROJECT_PATH / "figures"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# 2. 模型结构
# 必须与训练时完全一致
# =========================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# =========================================================
# 3. 根据训练时配置重建验证集
# =========================================================
def build_val_dataset(config):
    transform = transforms.ToTensor()

    mnist_train = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        transform=transform,
        download=True
    )

    #和train_cli.py保持一致
    generator = torch.Generator().manual_seed(config["seed"])
    subset_indices = torch.randperm(
        len(mnist_train),
        generator=generator
    )[:config["subset_size"]]

    small_mnist = Subset(
        mnist_train,
        subset_indices.tolist()
    )

    train_dataset, val_dataset = random_split(
        small_mnist,
        [config["train_size"],config["val_size"]],
        generator=generator
    )

    return val_dataset

# =========================================================
# 4. 找错分样本
# =========================================================
def find_misclassified(
        model,
        dataloader,
        max_samples=5
):
    model.eval()
    wrong_samples = []

    sample_id = 0

    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images)
            probs = torch.softmax(logits,dim=1)
            confidence, preds = probs.max(dim=1)

            wrong_mask = preds != labels

            wrong_indices = torch.where(wrong_mask)[0]
            for index in wrong_indices:
                wrong_samples.append(
                     {
                        "sample_id":
                            sample_id
                            + index.item(),

                        "image":
                            images[index].cpu(),

                        "true_label":
                            labels[index].item(),

                        "pred_label":
                            preds[index].item(),

                        "confidence":
                            confidence[index].item(),
                    }
                )

                if (
                    len(wrong_samples)
                    >= max_samples
                ):
                    return wrong_samples
            sample_id += images.size(0)

    return wrong_samples

# =========================================================
# 5. 保存 CSV
# =========================================================
def save_csv(wrong_samples):

    csv_path = RESULTS_DIR / "w2_misclassified.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f,
                                fieldnames=[
                                    "sample_id",
                "true_label",
                "pred_label",
                "confidence",
                "observation",
                                ])
        writer.writeheader()

        for sample in wrong_samples:
            writer.writerow(
                {
                    "sample_id":
                        sample["sample_id"],

                    "true_label":
                        sample["true_label"],

                    "pred_label":
                        sample["pred_label"],

                    "confidence":
                        sample["confidence"],

                    "observation":
                        ""
                }
            )

        print(
        "CSV saved:",
        csv_path
    )

# =========================================================
# 6. 保存错分图片
# =========================================================
def save_figure(wrong_samples):
    fig, axes = plt.subplots(
        1,
        len(wrong_samples),
        figsize=(12,3),
        layout="constrained"
    )

    # 防止只找到1个样本时axes不是数组
    if len(wrong_samples) == 1:
        axes = [axes]

    for ax, sample in zip(axes, wrong_samples):
        image = sample["image"].squeeze(0).numpy()
        ax.imshow(image,cmap="gray")
        ax.set_title(
            f"true={sample['true_label']}\n"
            f"pred={sample['pred_label']}\n"
            f"conf={sample['confidence']:.2f}"
        )
        ax.axis("off")

    figure_path = (
        FIGURES_DIR
        / "w02_misclassified.png"
    )
    plt.savefig(figure_path,dpi=150)
    plt.close()
    print(
        "figure saved:",
        figure_path
    )

# =========================================================
# 7. main
# =========================================================
def main():
    # -------------------------
    # 加载 checkpoint
    # -------------------------
    checkpoint = torch.load(
        CHECKPOINT_DIR,
        map_location="cpu"
    )
    CONFIG = checkpoint["config"]

    # -------------------------
    # 恢复模型，不需要训练因此只重建model
    # -------------------------
    model = MLP()
    model.load_state_dict(checkpoint["model_state_dict"])

    # -------------------------
    # 重建验证集
    # -------------------------
    val_dataset = build_val_dataset(CONFIG)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

    # -------------------------
    # 找5个错分样本
    # -------------------------
    wrong_samples = find_misclassified(
        model,val_dataloader
    )

    # -------------------------
    # 保存结果
    # -------------------------
    save_csv(wrong_samples)
    save_figure(wrong_samples)

if __name__ == "__main__":
    main()