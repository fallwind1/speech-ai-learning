import torch

from src.attention import dot_product_attention,make_causal_mask,make_padding_mask

def test_causal_future_change_does_not_affect_past():
    torch.manual_seed(42)

    B = 2
    T = 4
    d = 8

    x = torch.randn(B,T,d)

    causal_mask = make_causal_mask(T)
    split = 3

    x_after = x.clone()
    x_after[:,split:,:] = x_after[:,split:,:]*100+50

    output_before,_ = dot_product_attention(x,x,x,True,causal_mask)
    output_after,_ = dot_product_attention(x_after,x_after,x_after,True,causal_mask)

    assert torch.allclose(
        output_before[:,:split,:],
        output_after[:,:split,:],
        atol=1e-6
    )

def test_causal_future_weights_are_zero():
    torch.manual_seed(42)

    B,T,d = 2,4,8
    x = torch.randn(B,T,d)
    causal_mask = make_causal_mask(T)

    _,weights = dot_product_attention(x,x,x,True,valid_mask=causal_mask)

    invalid_positions = (~causal_mask).unsqueeze(0).expand_as(weights)

    assert torch.all(weights[invalid_positions] == 0)

def test_padding_weights_are_zero():
    torch.manual_seed(42)

    B,T,d = 2,4,8
    x = torch.randn(B,T,d)
    lengths = torch.Tensor([4,2])

    padding_mask = make_padding_mask(lengths,max_len=T).unsqueeze(1)
    _, weights = dot_product_attention(x,x,x,True,valid_mask=padding_mask)

    assert torch.all(weights[1,:,2:] == 0)

def test_padding_change_does_not_affect_valid_output():
    torch.manual_seed(42)
    
    B,T,d = 2,4,8
    x = torch.randn(B,T,d)
    lengths = torch.Tensor([4,2])

    padding_mask = make_padding_mask(lengths,max_len=T).unsqueeze(1)

    #修改padding项的K值，有效结果应当不变
    x_changed = x.clone()
    x_changed[1,2:,:] = x_changed[1,2:,:]*100+50

    output_before,_ = dot_product_attention(x,x,x,valid_mask=padding_mask)
    output_after,_ = dot_product_attention(x_changed,x_changed,x_changed,valid_mask=padding_mask)

    assert torch.allclose(
        output_before[1,:2,:],
        output_after[1,:2,:],
        atol=1e-6
    )

def test_all_masked_row_is_safe():
    torch.manual_seed(42)
        
    B,T,d = 2,4,8
    x = torch.randn(B,T,d)
    
    full_mask = torch.zeros(T,T,dtype=torch.bool)

    output,weights = dot_product_attention(x,x,x,valid_mask=full_mask)

    assert torch.isfinite(weights).all()
    assert torch.isfinite(output).all()

    assert torch.all(weights == 0)
    assert torch.all(output==0)