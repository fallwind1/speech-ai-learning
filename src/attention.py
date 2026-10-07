import math
import torch

def dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale:bool
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

def main():
    torch.manual_seed(42)
    B, T, d = 2, 4, 64
    Q = torch.randn(B, T, d)
    K = torch.randn(B, T, d)
    V = torch.randn(B, T, d)
    output_scale, weights_scale = scaled_dot_product_attention(Q,K,V,True)
    output_no_scale, weights_no_scale = scaled_dot_product_attention(Q,K,V,False)

    entropy_scale = attention_entropy(weights_scale)
    entropy_no_scale = attention_entropy(weights_no_scale)

    print(f"scaled:{entropy_scale}, no_scaled:{entropy_no_scale}")


if __name__ == "__main__":
    main()