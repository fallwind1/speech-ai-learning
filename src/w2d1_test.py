import torch

a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
c = torch.tensor(2.0, requires_grad=True)

d = a*b
L = d+c

L.backward()

print("a.grad=",a.grad)
print("b.grad =", b.grad)
print("c.grad =", c.grad)