import math
import torch

def dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale:bool = True,
        valid_mask=None
):
    #1.先做shape检查
    ## Q与K的特征维度必须相同
    assert Q.size(-1) == K.size(-1)

    ## KV序列长度必须相同
    assert K.size(-2) == V.size(-2)

    ## Batch-size一致
    assert Q.size(0) == K.size(0) == V.size(0)

    #2.计算scores
    scores = Q @ K.transpose(-2, -1)
    if scale:
        scores = scores / math.sqrt(Q.size(-1))

    #额外加入，如果输入valid_mask张量矩阵，那么进行mask操作
    if valid_mask is not None:
        assert valid_mask.dtype == torch.bool

        valid_mask = torch.broadcast_to(
            valid_mask,
            scores.shape,
        )

        masked_scores = scores.masked_fill(
            ~valid_mask,float("-inf")
        )

        valid_rows = valid_mask.any(
            dim=-1,keepdim=True
        )

        safe_scores = torch.where(
            valid_rows,
            masked_scores,
            torch.zeros_like(masked_scores)
        )

        weights = torch.softmax(safe_scores,dim=-1)

        weights = torch.where(
            valid_rows,
            weights,
            torch.zeros_like(weights)
        )
    else:
        #3.weights
        weights = torch.softmax(scores,dim=-1)

    #4.计算output
    output = weights @ V

    return output, weights

def attention_entropy(weights):
    weights = weights.clamp_min(1e-12)      #防止数值过低导致出现-inf，因此将小于1e-12的值均设为1e-12
    entropy = -(weights * torch.log(weights)).sum(dim=-1)
    entropy_mean = entropy.mean()

    return entropy_mean

def make_causal_mask(
        T: int,
        device = None
)->torch.Tensor:
    return torch.tril(
        torch.ones(
            T,T,
            dtype=torch.bool,
            device=device
        )
    )

def make_padding_mask(
        lengths: torch.Tensor,  #每个batch中样本的长度
        max_len: int            #最大长度
)->torch.Tensor:
    position = torch.arange(
        max_len,
        device=lengths.device
    ).unsqueeze(0)

    return position < lengths.unsqueeze(1)

def main():
    torch.manual_seed(42)
    B, T, d = 2, 4, 64
    Q = torch.randn(B, T, d)
    K = torch.randn(B, T, d)
    V = torch.randn(B, T, d)
    output_scale, weights_scale = dot_product_attention(Q,K,V,True)
    output_no_scale, weights_no_scale = dot_product_attention(Q,K,V,False)

    entropy_scale = attention_entropy(weights_scale)
    entropy_no_scale = attention_entropy(weights_no_scale)

    print(f"scaled:{entropy_scale}, no_scaled:{entropy_no_scale}")

#测试mask
def test_mask():
    torch.manual_seed(42)

    B, T, d = 2, 4, 8

    Q = torch.randn(B, T, d)
    K = torch.randn(B, T, d)
    V = torch.randn(B, T, d)
    
    causal_mask = make_causal_mask(T,device=Q.device)

    lengths = torch.tensor([4,2])
    padding_mask = make_padding_mask(lengths,max_len=4)

    combined_mask = causal_mask.unsqueeze(0) & padding_mask.unsqueeze(1)

    output, weights = dot_product_attention(Q,K,V,True,valid_mask=combined_mask)

    print("weights:")
    print(weights)

    print("row sums:")
    print(weights.sum(dim=-1))

    print(
        "weights finite:",
        torch.isfinite(weights).all()
    )

    print(
        "output finite:",
        torch.isfinite(output).all()
    )

    full_mask = torch.zeros_like(combined_mask,dtype=torch.bool)
    output0, weights0 = dot_product_attention(Q,K,V,True,valid_mask=full_mask)
    print(weights0)
    print(output0)


if __name__ == "__main__":
    test_mask()