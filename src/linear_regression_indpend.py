import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SEED = 42

figure_dir = Path("figures")
figure_dir.mkdir(
    parents=True,
    exist_ok=True
)

#1. data
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])
y = torch.tensor([  #y=2x+4
    [6.0],
    [8.0],
    [10.0],
    [12.0],
    [14.0]
])

###确认数据类型及规模，以及数据在哪计算
"""
print("x shape:",x.shape)
print("y shape:",y.shape)
print("x dtype:",x.dtype)
print("y dtype:",y.dtype)
print("x device",x.device)
print("y device",y.device)
"""

#4. train loop function
def train_loop(use_zero_grad: bool) -> list[float]:

    print("use_zero_grad:{use_zero_grad}")

    torch.manual_seed(SEED)

    #2. model
    model = torch.nn.Linear(1,1)

    #3. loss & optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = 0.08
    )
    loss_history = []

    for epoch in range(200):
        if use_zero_grad:
            optimizer.zero_grad()
        if epoch < 3:
            print(
                "before backward:",
                model.weight.grad
            )

        y_pred = model(x)   #forward
        loss = criterion(y_pred,y)
        loss.backward()
        
        if epoch < 3:
            print(
                    "after backward:",
                    model.weight.grad
                    )

        optimizer.step()

        if epoch % 10 == 0: #每10轮记录结果
            loss_history.append(loss.item())
            """
            print(
                f"epoch:{epoch}",
                f"loss:{loss:.6f}",
                f"weight:{model.weight.item():.4f}",
                f"bias:{model.bias.item():.4f}"
            )
            """
    #利用最终结果预测值
    with torch.no_grad():
        final_pred = model(x).detach().numpy()
    
    print(f"final weight:{model.weight.item():.4f}, final bias:{model.bias.item():.4f}")
    return loss_history, final_pred

#进行三组实验
loss_history1,pred1 = train_loop(True)
loss_history2,pred2 = train_loop(False)
loss_history3,pred3 = train_loop(True)

#5. 可视化loss-epoch图
fig, axes = plt.subplots(1,2,figsize=(10,4), layout="constrained")
axes[0].plot(range(0,200,10),loss_history1,label="normal")
axes[0].plot(range(0,200,10),loss_history2,label="no zero_grad")
axes[0].plot(range(0,200,10),loss_history3,label="fixed")
axes[0].set(title="Training curve", xlabel="Epoch", ylabel="Loss")
axes[0].legend()
axes[1].plot(x.detach().numpy(),pred1,label="pred1")
axes[1].plot(x.detach().numpy(),pred2,label="pred2")
axes[1].plot(x.detach().numpy(),pred3,label="pred3")
axes[1].set(title="pred curve", xlabel="x", ylabel="y_pred")
axes[1].legend()
plt.tight_layout()
plt.savefig(
    "figures/w01_loss,png",
    dpi=600
)
plt.show()