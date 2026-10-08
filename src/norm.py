import torch
from torch import nn

class ManualLayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()

        self.eps = eps

        #初始化可学习参数
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True,unbiased=False)

        x_norm = (x-mean) / torch.sqrt(var + self.eps)

        y = self.gamma * x_norm + self.beta
        return y

def main():
    torch.manual_seed(42)

    B,T,d = 2,3,4
    x = torch.randn(B,T,d)

    manual = ManualLayerNorm(d)
    reference = nn.LayerNorm(d)

    manual_output = manual(x)
    reference_output = reference(x)

    max_error = (manual_output-reference_output).abs().max()
    is_close = torch.allclose(
        manual_output,
        reference_output,
        atol=1e-6,
        rtol=1e-5
    )

    print("Manual output shape",manual_output.shape)
    print("reference output shape",reference_output.shape)

    print("最大绝对值误差：",max_error)
    print("是否近似相等：",is_close)

    with torch.no_grad():
        gamma = torch.tensor([1.0, 2.0, 0.5, 1.5])
        beta = torch.tensor([0.1, -0.2, 0.3, 0.0])

        manual.gamma.copy_(gamma)
        manual.beta.copy_(beta)

        reference.weight.copy_(gamma)
        reference.bias.copy_(beta)

    manual_output = manual(x)
    reference_output = reference(x)

    max_error = (manual_output-reference_output).abs().max()
    is_close = torch.allclose(
            manual_output,
            reference_output,
            atol=1e-6,
            rtol=1e-5
        )
    print("最大绝对值误差：",max_error)
    print("是否近似相等：",is_close)

if __name__ == "__main__":
    main()