from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

#设置参数
SEED = 42
SUBSET_SIZE = 2000
DATA_DIR = Path("data")
TRAIN_SIZE = 1600
VAL_SIZE = 400
BATCH_SIZE = 64

#1. 取MNIST中全部train数据集
transform = transforms.ToTensor()

mnist_train = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    transform=transform,
    download=True
)

print("Dataset size:", len(mnist_train))
image, label = mnist_train[0]

#2. 构建小型子集
subset_generator = torch.Generator().manual_seed(SEED)  #固定随机生成器种子

subset_indices = torch.randperm(
    len(mnist_train),
    generator=subset_generator,
)[:SUBSET_SIZE]

small_mnist = Subset(
    mnist_train,
    subset_indices.tolist()
)

print(
    "Small dataset size:",
    len(small_mnist),
)

#3. 将小型子集划分为train和vaild
split_generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    small_mnist,
    [TRAIN_SIZE, VAL_SIZE],
    generator=split_generator
)

train_indices = set(train_dataset.indices)
val_indices = set(val_dataset.indices)
overlap = train_indices & val_indices

print("Overlap:",len(overlap))

#创建DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

images,  labels = next(     #next(...)取第一个batch
    iter(train_loader)  #创建一个可以不断获取喜爱个batch的迭代器
)

print(
    "images shape:",
    images.shape,
)

print(
    "images dtype:",
    images.dtype,
)

print(
    "labels shape:",
    labels.shape,
)

print(
    "labels dtype:",
    labels.dtype,
)