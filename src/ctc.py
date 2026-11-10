import torch

def extend_target(target, blank=0):
    """
    将原始目标序列扩展为 CTC 目标序列。

    参数：
        target: 原始目标 Token ID 序列，不包含 blank
        blank: Blank 对应的 Token ID

    返回：
        extended_target: 扩展目标序列
    """
    target = [int(token) for token in target]

    # target 中不能包含 blank
    if blank in target:
        raise ValueError(
            "target must not contain blank"
        )

    #初始化扩展目标
    extended_target = [blank] * (
        2 * len(target) + 1
    )

    #将原始token放入奇数位置
    extended_target[1::2] = target

    return extended_target

def validate_ctc_inputs(
        log_probs,
        target,
        blank=0
):
    """
    检查 CTC 前向算法输入是否合法。
    """
    if not isinstance(log_probs, torch.Tensor):
        raise TypeError(
            "log_probs must be a torch.Tensor"
        )

    if log_probs.ndim != 2:
        raise ValueError(
            "log_probs must have shape (T, C)"
        )

    if not log_probs.is_floating_point():
        raise TypeError(
            "log_probs must use floating dtype"
        )

    T, C = log_probs.shape

    if C == 0:
        raise ValueError(
            "vocabulary size must be positive"
        )

    blank = int(blank)

    if not 0 <= blank < C:
        raise ValueError(
            "blank ID is out of vocabulary range"
        )

    target = [int(token) for token in target]

    for token in target:
        if not 0 <= token < C:
            raise ValueError(
                f"Invalid target token ID: {token}"
            )

        if token == blank:
            raise ValueError(
                "target must not contain blank"
            )

    return target, blank

def ctc_forward_log(
        log_probs,
        target,
        blank=0
):
    """
    使用 Log 空间前向算法计算 CTC 目标概率。

    参数：
        log_probs:
            Tensor，形状为 (T, C)
            已经过 Log Softmax

        target:
            原始目标 Token ID 序列
            不包含 Blank

        blank:
            Blank 对应的 Token ID

    返回：
        scalar Tensor:
            log P(target | input)
    """
    #1.输入合法性检查
    target, blank = validate_ctc_inputs(
        log_probs,
        target,
        blank
    )

    T, C = log_probs.shape
    U = len(target)

    #2.空目标特殊处理
    if U == 0:
        if T == 0:
            return log_probs.new_zeros(())

        return log_probs[:,blank].sum()

    #3.时间帧为空，但目标非空
    if T == 0:
        return log_probs.new_full(
            (),
            float("-inf")
        )

    #4.构造扩展目标
    extended_target = extend_target(
        target,
        blank
    )

    S = len(extended_target)

    #5.初始化第一行
    neg_inf = log_probs.new_tensor(
        float("-inf")
    )

    first_row = [neg_inf] * S

    #状态0：Blank
    first_row[0] = log_probs[0, blank]

    #状态1：第一个目标Token
    first_row[1] = log_probs[
        0, extended_target[1]
    ]

    rows = [torch.stack(first_row)]

    #6.前向递推
    for t in range(1, T):
        prev = rows[-1]
        current = []

        for s in range(S):

            #stay
            predecessors = [prev[s]]

            #advance
            if s >= 1:
                predecessors.append(
                    prev[s-1]
                )

            #skip
            if (
                s>=2
                and extended_target[s] != blank
                and extended_target[s] != extended_target[s-2]
            ):
                predecessors.append(prev[s-2])

            #LogSumExp
            prev_sum = torch.logsumexp(
                torch.stack(predecessors),
                dim=0
            )

            #当前时间帧的发射概率
            current_value = (
                log_probs[t, extended_target[s]]
                + prev_sum
            )

            current.append(current_value)

        rows.append(torch.stack(current))

    #7.构造完整DP矩阵
    alpha = torch.stack(rows)

    #8.终止状态求和
    log_likelihood = torch.logsumexp(
        alpha[-1, S-2:S],
        dim=0
    )

    return log_likelihood