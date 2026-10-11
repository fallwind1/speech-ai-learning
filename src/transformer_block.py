import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )

    def forward(self,x):
        #attention residual
        residual = x
        x_norm = self.norm1(x)

        attn_output,attn_weights = self.attn(x_norm,x_norm,x_norm)

        x = attn_output + residual

        #ffn residual
        residual = x

        x_norm = self.norm2(x)

        x = residual + self.ffn(x_norm)

        return x, attn_weights

def main():
    torch.manual_seed(42)
    B,T,d = 2,5,32
    x = torch.randn(B,T,d)
    print(f"input x.shape:{x.shape}")

    block = TransformerBlock(d,4,128)

    output,weights = block(x)
    print(f"output shape:{output.shape}")
    print(f"attention shape:{weights.shape}")

    loss = output.mean()
    loss.backward()

    for name,param in block.named_parameters():
        print(name,param.grad is not None)

if __name__ == "__main__":
    main()