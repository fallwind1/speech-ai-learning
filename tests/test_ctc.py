# ==============================
# 1. 导入
# ==============================

from itertools import product

import torch
import sys
from pathlib import Path

root = Path(r"D:\Projects\speech-ai-learning")
if root not in sys.path:
    sys.path.insert(0, str(root))

from src.ctc import ctc_forward_log

def ctc_collapse(
        path,
        blank=0
):
    """
    将一条 CTC 路径转换为最终 Token 序列。

    步骤：
    1. 合并相邻重复 Token
    2. 删除 Blank
    """

    merged = []

    previous = None

    #第一步：合并重复token
    for token in path:
        if token != previous:
            merged.append(token)

        previous = token

    #第二部：删除blank
    result = [
        token
        for token in merged
        if token != blank
    ]

    return result

def ctc_bruteforce_log(
    log_probs,
    target,
    blank=0
):
    """
    使用暴力枚举计算 CTC 目标序列的 Log 概率。

    参数：
        log_probs:
            Tensor，形状为 (T, C)
            已经过 Log Softmax

        target:
            原始目标 Token 序列

        blank:
            Blank 对应的 Token ID

    返回：
        scalar Tensor:
            log P(target | input)
    """

    #1.获取时间帧数量和词表大小
    T, C = log_probs.shape

    target = [
        int(token)
        for token in target
    ]

    #2.枚举所有可能路径
    path_scores = []

    for path in product(
        range(C),
        repeat=T
    ):
        #3.执行CTC Collapse
        collapsed = ctc_collapse(
            path,
            blank
        )

        #4.判断是否与目标序列一致
        if list(collapsed) != target:
            continue

        #5.计算当前路径的Log概率
        path_score = log_probs.new_zeros(())

        for t, token in enumerate(path):

            path_score = (
                path_score
                + log_probs[t, token]
            )

        path_scores.append(path_score)

    #6. 没有合法路径
    if len(path_scores) == 0:

        return log_probs.new_full(
            (),
            float("-inf")
        )

    # 7. 对所有合法路径进行 LogSumExp
    return torch.logsumexp(
        torch.stack(path_scores),
        dim=0
    )

# ==============================
# 4. 普通目标测试
# ==============================

def test_bruteforce_matches_dp():
    probs = torch.tensor(
        [
            [0.6, 0.4],
            [0.3, 0.7]
        ],
        dtype=torch.float32
    )

    log_probs = torch.log(probs)

    target = [1]

    # 暴力枚举
    brute_result = ctc_bruteforce_log(
        log_probs,
        target,
        blank=0
    )

    # 动态规划
    dp_result = ctc_forward_log(
        log_probs,
        target,
        blank=0
    )

    #Pytorch ctcloss
    torch_loss = torch_ctc_loss_one(
        log_probs,
        target,
        blank=0
    )

    print("Brute Force:", brute_result.item())
    print("Dynamic Programming:", dp_result.item())
    print("PyTorch CTCLoss:", torch_loss.item())
    print("Negative CTCLoss:", (-torch_loss).item())

    assert torch.allclose(
        brute_result,
        dp_result,
        atol=1e-6
    )

    assert torch.allclose(
        brute_result.exp(),
        torch.tensor(0.82),
        atol=1e-6
    )

    print("CTC enumeration test passed!")


# ==============================
# 5. 重复 Token 测试
# ==============================

def test_repeated_tokens():
    target = [1,1]
    probs = torch.tensor([
        [0.5,0.5],
        [0.5,0.5],
        [0.5,0.5]
    ],
    dtype=torch.float32
    )

    log_probs = torch.log(probs)

    brute_result = ctc_bruteforce_log(
        log_probs,
        target,
        blank=0
    )

    dp_result = ctc_forward_log(
        log_probs,
        target,
        blank=0
    )

    assert torch.allclose(
        brute_result,
        dp_result,
        atol=1e-6
    )

    assert torch.allclose(
        brute_result.exp(),
        torch.tensor(0.125),
        atol=1e-6
    )

    print("Repeated token test passed!")

# ==============================
# 6. 空目标测试
# ==============================

def test_empty_target():
    target = []
    probs = torch.tensor([
            [0.5,0.5],
            [0.5,0.5],
            [0.5,0.5]
        ],
        dtype=torch.float32
    )
    log_probs = torch.log(probs)
    
    brute_result = ctc_bruteforce_log(
        log_probs,
        target,
        blank=0
    )

    dp_result = ctc_forward_log(
        log_probs,
        target,
        blank=0
    )

    #Pytorch ctcloss
    torch_loss = torch_ctc_loss_one(
        log_probs,
        target,
        blank=0
    )

    assert torch.allclose(
            brute_result.exp(),
            torch.tensor(0.125),
            atol=1e-6
        )


# ==============================
# 7. 不可行对齐测试
# ==============================

def test_impossible_alignment():
    target = [1,1]
    probs = torch.tensor([
            [0.5,0.5],
            [0.5,0.5]
        ],
        dtype=torch.float32
    )
    log_probs = torch.log(probs)
    
    brute_result = ctc_bruteforce_log(
        log_probs,
        target,
        blank=0
    )

    dp_result = ctc_forward_log(
        log_probs,
        target,
        blank=0
    )

    #Pytorch ctcloss
    torch_loss = torch_ctc_loss_one(
        log_probs,
        target,
        blank=0
    )

    assert torch.allclose(
                brute_result.exp(),
                torch.tensor(0,dtype=torch.float),
                atol=1e-6
            )


def torch_ctc_loss_one(
    log_probs,
    target,
    blank=0
):
    """
    使用 PyTorch CTCLoss 计算单条样本的 CTC Loss。

    参数：
        log_probs: (T, C)，Log 概率
        target: 原始目标 Token 序列
        blank: Blank ID

    返回：
        标量 Tensor，负 Log 概率
    """

    T, C = log_probs.shape

    #增加Batch维度
    log_probs_batch = log_probs.unsqueeze(1)

    #构造目标序列
    targets = torch.tensor(
        target,
        dtype=torch.long,
        device=log_probs.device
    )

    # 输入长度
    input_lengths = torch.tensor(
        [T],
        dtype=torch.long
    )

    # 目标长度
    target_lengths = torch.tensor(
        [len(target)],
        dtype=torch.long
    )

    #CTCLoss
    criterion = torch.nn.CTCLoss(
        blank=blank,
        reduction="none",
        zero_infinity=False
    )

    loss = criterion(
        log_probs_batch,
        targets,
        input_lengths,
        target_lengths
    )

    return loss[0]