def build_dp(reference_tokens, hypothesis_tokens):
    """
    构建最小编辑距离 DP 矩阵。
    """
    m = len(reference_tokens)
    n = len(hypothesis_tokens)

    dp = [
        [0]*(n+1)
        for _ in range(m+1)
    ]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):

            cost = (
                0
                if reference_tokens[i-1] == hypothesis_tokens[j-1]
                else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,         # D
                dp[i][j - 1] + 1,         # I
                dp[i - 1][j - 1] + cost,  # C / S
            )

    return dp

def backtrace(reference_tokens, hypothesis_tokens, dp):
    """
    沿着最优路径回溯，返回编辑操作列表。

    每个操作为：
    (操作类型, Reference token, Hypothesis token)
    """
    i = len(reference_tokens)
    j = len(hypothesis_tokens)

    operations = []

    while i>0 or j>0:

        #1.正确匹配：优先选择
        if (
            i>0
            and j>0
            and reference_tokens[i-1] == hypothesis_tokens[j-1]
            and dp[i][j] == dp[i-1][j-1]
        ):
            operations.append((
                "C",
                reference_tokens[i-1],
                hypothesis_tokens[j-1]
            ))
            i -= 1
            j -= 1

        #2. 替换
        elif (
            i > 0
            and j > 0
            and dp[i][j] == dp[i-1][j-1] + 1
        ):
            operations.append((
                "S",
                reference_tokens[i-1],
                hypothesis_tokens[j-1]
            ))
            i -= 1
            j -= 1

        #3.删除
        elif (
            i>0
            and dp[i][j] == dp[i-1][j] + 1
        ):
            operations.append((
                "D",
                reference_tokens[i-1],
                None
            ))
            i -= 1

        #4.插入
        elif (
            j > 0
            and dp[i][j] == dp[i][j-1] + 1
        ):
            operations.append((
                "I",
                None,
                hypothesis_tokens[j-1]
            ))
            j -= 1

        else:
            raise RuntimeError("DP回溯失败")

    #回溯是从末尾到开头，需要反转
    operations.reverse()

    return operations

def score_tokens(reference_tokens, hypothesis_tokens):
    """
    计算 token 级编辑距离、S/D/I/C 和错误率。
    """
    dp = build_dp(reference_tokens,hypothesis_tokens)

    operations = backtrace(
        reference_tokens,
        hypothesis_tokens,
        dp
    )

    counts = {
        "S": 0,
        "D": 0,
        "I": 0,
        "C": 0
    }

    for op, _, _ in operations:
        counts[op] += 1

    distance = dp[-1][-1]
    N = len(reference_tokens)

    assert (
        counts["S"] + counts["D"] + counts["I"]
        == distance
    )

    #空reference的特殊处理在模块3继续讨论
    error_rate = distance / N if N > 0 else None

    return {
        "S": counts["S"],
        "D": counts["D"],
        "I": counts["I"],
        "C": counts["C"],
        "N": N,
        "distance": distance,
        "error_rate": error_rate,
        "operations": operations,
    }

def score_corpus(token_pairs):
    """
    计算语料级错误率。

    token_pairs:
        [
            (reference_tokens, hypothesis_tokens),
            ...
        ]

    返回：
        各类操作总数、参考token总数、
        总编辑距离与语料级错误率。
    """

    totals = {
        "S": 0,
        "D": 0,
        "I": 0,
        "C": 0,
        "N": 0,
        "distance": 0,
    }

    sample_results = []

    for reference_tokens, hypothesis_tokens in token_pairs:

        result = score_tokens(reference_tokens,hypothesis_tokens)

        sample_results.append(result)

        for key in totals:
            totals[key] += result[key]

    #使用总编辑数 / 总参考token数
    if totals["N"] > 0:
        error_rate = (
            totals["distance"] / totals["N"]
        )
    else:
        error_rate = None

    assert (
        totals["I"] + totals["D"] + totals["S"]
        == totals["distance"]
    )

    return {
        **totals,
        "error_rate": error_rate,
        "sample_results": sample_results,
    }