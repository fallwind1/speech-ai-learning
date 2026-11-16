import torch
import sys
from pathlib import Path

root = Path(r"D:\Projects\speech-ai-learning")
if root not in sys.path:
    sys.path.insert(0, str(root))

from src.conformer import ConformerConvModule

model = ConformerConvModule(dim=64,kernel_size=31)

x = torch.randn(2,100,64,requires_grad=True)

y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)

# 检查梯度
loss = y.mean()
loss.backward()

print("Input gradient exists:", x.grad is not None)

# 检查 Depthwise Conv 参数量
print("Depthwise weight shape:", model.depthwise.weight.shape)

params = sum(
    p.numel() for p in model.depthwise.parameters()
)

print("Depthwise parameters:", params)