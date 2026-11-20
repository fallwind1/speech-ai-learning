import torch.nn as nn
import torch
from pathlib import Path
import sys

root = Path(r"D:\Projects\speech-ai-learning")
if root not in sys.path:
    sys.path.insert(0, str(root))

from src.attention import dot_product_attention, make_padding_mask


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

class ConformerFFN(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()

        self.layernorm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * expansion_factor)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * expansion_factor, dim)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.linear1(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x

class ConformerMHSA(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # LayerNorm
        self.layernorm = nn.LayerNorm(dim)
        # Q/K/V Linear
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        # Output Linear
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_mask=None):
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        # 1. LayerNorm
        h = self.layernorm(x)
        # 2. Q/K/V 投影
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        # 3. 拆分 Head
        q = q.reshape(B,T,H,D).transpose(1,2)
        k = k.reshape(B, T, H, D).transpose(1, 2)
        v = v.reshape(B, T, H, D).transpose(1, 2)
        # 4. 扩展 valid_mask
        attn_mask = None
        if valid_mask is not None:
            assert valid_mask.shape == (B, T)
            attn_mask = valid_mask[:, None, None, :]
        # 5. 调用 dot_product_attention
        out, weights = dot_product_attention(
            q, k, v,
            valid_mask=attn_mask
        )
        # 6. 合并 Head
        out = out.transpose(1, 2)
        out = out.contiguous().reshape(B, T, C)
        # 7. Output Linear
        out = self.out_proj(out)
        out = self.dropout(out)

        return out

class ConformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=4,
            ff_expansion=4,
            kernel_size=31,
            dropout=0.1
    ):
        super().__init__()

        #1.创建第一个ConformerFFN
        self.ffn1 = ConformerFFN(dim,ff_expansion,dropout)

        #2.创建ConformerMHSA
        self.mhsa = ConformerMHSA(dim,num_heads,dropout)

        #3：复用 ConformerConvModule
        self.conv = ConformerConvModule(dim,kernel_size,dropout)

        # TODO 4：创建第二个 ConformerFFN
        self.ffn2 = ConformerFFN(dim,ff_expansion,dropout)

        # TODO 5：创建 Final LayerNorm
        self.layernorm = nn.LayerNorm(dim)

    def zero_padding(self, x, valid_mask):
        if valid_mask is None:
            return x

        return x.masked_fill(
            ~valid_mask.unsqueeze(-1),
            0.0
        )

    def forward(self,x,valid_mask):
        x = self.zero_padding(x, valid_mask)

        #1：第一半步 FFN + Residual
        x = x + 0.5 * self.ffn1(x)
        x = self.zero_padding(x, valid_mask)

        #2：MHSA + Residual
        x = x + self.mhsa(x, valid_mask)
        x = self.zero_padding(x, valid_mask)

        #3：ConvModule + Residual
        x = x + self.conv(x)
        x = self.zero_padding(x, valid_mask)

        #4：第二半步 FFN + Residual
        x = x + 0.5 * self.ffn2(x)
        x = self.zero_padding(x, valid_mask)

        #5：Final LayerNorm
        x = self.layernorm(x)
        x = self.zero_padding(x, valid_mask)

        return x

    