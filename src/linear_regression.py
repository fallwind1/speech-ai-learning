import torch
from torch import nn

#1.生成数据
x = torch.tensor([
    [0.0],
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])
y = torch.tensor([
    [2.0],
    [5.0],
    [8.0],
    [11.0],
    [14.0]
])

#2.创建模型 & 优化器 &损失函数
model = nn.Linear(1,1) #in_features, out_features
criterion = nn.MSELoss()    #默认mean
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)
# 查看模型初始参数
print("Initial weight:",model.weight.item())
print("Initial bias:",model.bias.item())

#3.训练循环
for step in range(100):
    optimizer.zero_grad()
    #1.forward计算
    y_pred = model(x)

    #2.loss
    loss = criterion(y_pred, y)

    #3.backward
    loss.backward()

    #4. update parameters
    optimizer.step()

    if step % 10 == 0:
        print(
            f"step:{step}",
            f"loss={loss.item():.6f}",
            f"weight={model.weight.item():.4f}",
            f"bias={model.bias.item():.4f}"
        )

print("\nFinal weight:", model.weight.item())
print("Final bias:", model.bias.item())

#4.任取数据测试
test_data = torch.tensor([[5.0]])
with torch.no_grad():
    test_pred = model(test_data)
print("Prediction for x=5:", test_pred.item())
print("Expected value:", 17.0)