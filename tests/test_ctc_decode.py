import sys
from pathlib import Path

root = Path(r"D:\Projects\speech-ai-learning")
if root not in sys.path:
    sys.path.insert(0, str(root))

from src.ctc_decode import (
    ctc_collapse,
    ctc_greedy_decode,
)

import torch


def test_repeated_tokens():
    """
    相邻重复 Token 应合并。
    """

    path = [1,1,0,2,2]

    assert ctc_collapse(path, blank=0) == [1,2]

def test_blank_separates_repeated_tokens():
    """
    Blank 可以分隔相同 Token。
    这是原 W8 错误折叠例子的回归测试。
    """
    path = [1, 0, 1, 2]

    assert ctc_collapse(path, blank=0) == [1, 1, 2]

def test_consecutive_blanks():
    """
    连续 Blank 应全部删除。
    """
    path = [1, 0, 0, 2]

    assert ctc_collapse(path, blank=0) == [1, 2]


def test_all_blank():
    """
    全 Blank Path 应得到空序列。
    """
    path = [0, 0, 0]

    assert ctc_collapse(path, blank=0) == []


def test_empty_path():
    """
    空 Path 应得到空序列。
    """
    assert ctc_collapse([], blank=0) == []


def test_greedy_decode():
    """
    验证逐帧 argmax 和 CTC Collapse。
    """
    log_probs = torch.tensor([
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])

    result = ctc_greedy_decode(
        log_probs,
        blank=0,
    )

    assert result == [1, 1, 2]