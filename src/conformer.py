import torch.nn as nn
import torch

class ConformerConvModule(nn.Module):
    def __init__(self,dim,kernel_size=31,dropout=0.1):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.layernorm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Conv1d(dim, 2*dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim,
            bias=False
        )
        self.batchnorm = nn.BatchNorm1d(dim)
        self.silu = nn.SiLU()
        self.pointwise2 = nn.Conv1d(dim,dim,kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. LayerNorm
        x = self.layernorm(x)

        # 2. (B,T,C) -> (B,C,T)
        x = x.transpose(1,2)

        # 3. Pointwise Conv1
        x = self.pointwise1(x)

        # 4. GLU
        x = self.glu(x)

        # 5. Depthwise Conv
        x = self.depthwise(x)

        # 6. BatchNorm
        x = self.batchnorm(x)

        # 7. SiLU
        x = self.silu(x)

        # 8. Pointwise Conv2
        x = self.pointwise2(x)

        # 9. Dropout
        x = self.dropout(x)

        # 10. (B,C,T) -> (B,T,C)
        x = x.transpose(1,2)

        return x

