import torch

def ctc_collapse(path, blank=0):
    """
    对 CTC 路径进行折叠。

    先合并相邻重复 Token，
    再删除 Blank。
    """

    output = []
    previous = None

    for token in path:

        if token != previous and token != blank:
            output.append(token)

        previous = token

    return output

def ctc_greedy_decode(log_probs, blank=0):
    """
    Greedy CTC 解码。

    参数：
        log_probs: (T, C)
        blank: Blank ID

    返回：
        最终 Token 序列
    """

    if not isinstance(log_probs, torch.Tensor):
        raise TypeError("log_probs must be a torch.Tensor")

    if log_probs.ndim != 2:
        raise ValueError(
            "log_probs must have shape (T,C)"
        )

    if not 0<=blank<log_probs.shape[1]:
        raise ValueError("invalid blank ID")

    #逐帧选择最大概率Token
    best_path = (
        log_probs
        .argmax(dim=-1)
        .detach()
        .cpu()
        .tolist()
    )

    return ctc_collapse(
        best_path,
        blank=blank
    )