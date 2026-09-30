import torch
from torch import nn
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader,Subset,random_split
from torchvision import datasets, transforms
import random

# =========================================================
# 1. Config
# =========================================================
CONFIG = {
    "seed":42,
    "subset_size":2000,
    "train_size":1600,
    "val_size":400,
    "batch_size":64,
    "hidden_dim":128,
    "lr":0.1,
    "momentum":0.9,
    "epochs":5
}

CHECKPOINT_PATH = Path(
    "checkpoints/w2d4_checkpoint.pt"
)

# =========================================================
# 2. Reproducibility
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# =========================================================
# 3. Model
# =========================================================
class MLP(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()      #调用当前类的父类的 __init__() 方法，也就是初始化父类。

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,10)
        )

    def forward(self,x):
        return self.model(x)

# =========================================================
# 4. Dataset
# =========================================================
def build_dataset(config):
    full_dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    #随机抽取2000个
    subset_generator = torch.Generator()
    subset_generator.manual_seed(config["seed"])

    subset_indcies = torch.randperm(
        len(full_dataset),
        generator=subset_generator
    )[:config["subset_size"]]

    small_dataset = Subset(
        full_dataset,
        subset_indcies.tolist()
    )

    #固定train/val划分
    split_generator = torch.Generator()
    split_generator.manual_seed(
        config["seed"] + 1
    )
    train_dataset, val_dataset = random_split(
        small_dataset,
        [config["train_size"],config["val_size"]],
        generator=split_generator
    )

    return train_dataset, val_dataset

# =========================================================
# 5. DataLoader
# =========================================================
def make_train_loader(train_dataset,config,epoch):
    generator = torch.Generator()

    #同一个epoch永远得到相同shuffle顺序
    generator.manual_seed(
        config["seed"] + epoch
    )

    return DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0
    )

def make_val_loader(val_dataset,config):
    return DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )

# =========================================================
# 6. Training
# =========================================================
def train_one_epoch(model,dataloader,criterion,optimizer,device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits,labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected.")
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item()*batch_size
        total_samples += batch_size

    return total_loss / total_samples

# =========================================================
# 7. Validation
# =========================================================
def evaluate(model,dataloader,criterion,device):
    model.eval()

    total_loss =0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits,labels)
            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples

# =========================================================
# 8. Save checkpoint
# =========================================================

def save_checkpoint(path,model,optimizer,epoch,config):
    path.parent.mkdir(parents=True,exist_ok=True)
    
    checkpoint = {
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "epoch":epoch,
        "config":config,
        "seed":config["seed"],
        # 更严格恢复随机过程时使用
        "torch_rng_state":
            torch.get_rng_state(),
    }

    if torch.cuda.is_available():
            checkpoint[
                "cuda_rng_state_all"
            ] = torch.cuda.get_rng_state_all()

    torch.save(
        checkpoint,
        path
    )

# =========================================================
# 9. Load checkpoint
# =========================================================
def load_checkpoint(path,model,optimizer,device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # 恢复CPU RNG state
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(
            checkpoint["torch_rng_state"].cpu()
        )

    # 同环境CUDA恢复
    if(device.type == "cuda" and "cuda_rng_state_all" in checkpoint):
        cuda_rng_states = [
        state.cpu()
        for state
        in checkpoint["cuda_rng_state_all"]
    ]

    torch.cuda.set_rng_state_all(
        cuda_rng_states
    )

    start_epoch = checkpoint["epoch"] + 1

    return (start_epoch,checkpoint["config"],checkpoint["seed"])

# =========================================================
# 10. Parameter verification
# =========================================================
def clone_parameters(model):

    return {
        name: param.detach().cpu().clone()

        for name, param in model.named_parameters()
    }

def assert_parameters_equal(reference,model):
    for name, param in (model.named_parameters()):
        current = param.detach().cpu()

        assert torch.equal(
            reference[name],current
        ),(f"Parameter mismatch:{name}")

def assert_parameters_close(reference,model,rtol=1e-5,atol=1e-7):
    for name, param in model.named_parameters():
        current = param.detach().cpu()

        assert torch.allclose(
            reference[name],
            current,
            rtol=rtol,
            atol=atol
        ),(f"Trajectory mismatch:{name}")

# =========================================================
# 11. Main
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    set_seed(CONFIG["seed"])

    train_dataset, val_dataset = build_dataset(CONFIG)

    val_loader = make_val_loader(val_dataset,CONFIG)

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Phase A:
    # Train Epoch 0
    # -----------------------------------------------------
    model = MLP(CONFIG["hidden_dim"]).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG["lr"],
        momentum=CONFIG["momentum"]
    )

    epoch = 0

    train_loader = make_train_loader(
        train_dataset,
        CONFIG,
        epoch
    )

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    val_loss = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    print(
        f"Epoch {epoch}: "
        f"train={train_loss:.4f}, "
        f"val={val_loss:.4f}"
    )

    #保存时参数副本
    checkpoint_parameters = clone_parameters(model)

    save_checkpoint(
        CHECKPOINT_PATH,
        model,
        optimizer,
        epoch,
        CONFIG,
    )

    print(
        "Checkpoint saved:",
        CHECKPOINT_PATH,
    )

    # -----------------------------------------------------
    # Phase B:
    # 不中断继续训练 Epoch 1
    # 作为 reference
    # -----------------------------------------------------
    reference_epoch = 1

    train_loader = make_train_loader(
        train_dataset,
        CONFIG,
        reference_epoch,
    )

    reference_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    reference_parameters = (
        clone_parameters(model)
    )

    print(
        "Uninterrupted Epoch 1 loss:",
        f"{reference_loss:.4f}",
    )

    # -----------------------------------------------------
    # Phase C:
    # 创建全新的 model / optimizer
    # -----------------------------------------------------

    restored_model = MLP(CONFIG["hidden_dim"]).to(device)

    restored_optimizer = torch.optim.SGD(
        restored_model.parameters(),
        lr=CONFIG["lr"],
        momentum=CONFIG["momentum"]
    )

    # -----------------------------------------------------
    # Phase D:
    # Load checkpoint
    # -----------------------------------------------------
    (
        start_epoch,
        restored_config,
        restored_seed,
    ) = load_checkpoint(
        CHECKPOINT_PATH,
        restored_model,
        restored_optimizer,
        device,
    )

    print(
        "Resume from epoch:",
        start_epoch,
    )

    # -----------------------------------------------------
    # Phase E:
    # 严格验证：
    # checkpoint刚加载后的参数
    # -----------------------------------------------------
    assert_parameters_equal(
        checkpoint_parameters,
        restored_model,
    )
    print(
        "Checkpoint parameter test: PASS"
    )

    # -----------------------------------------------------
    # Phase F:
    # Resume Epoch 1
    # -----------------------------------------------------
    resumed_loader = make_train_loader(
        train_dataset,
        restored_config,
        start_epoch,
    )

    resumed_loss = train_one_epoch(
        restored_model,
        resumed_loader,
        criterion,
        restored_optimizer,
        device,
    )

    print(
        "Resumed Epoch 1 loss:",
        f"{resumed_loss:.4f}",
    )

    # -----------------------------------------------------
    # Phase G:
    # 比较后续训练轨迹
    # -----------------------------------------------------
    assert_parameters_close(
        reference_parameters,
        restored_model,
        rtol=1e-5,
        atol=1e-7,
    )

    print(
        "Resume trajectory test: PASS"
    )


if __name__ == "__main__":
    main()