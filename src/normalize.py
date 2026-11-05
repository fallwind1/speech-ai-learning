import re
import unicodedata

#W7D4 评分规则草案，模块3测试后再冻结
NORMALIZER_VERSION = "w7d4-v1-draft"


#本项目约定：这些普通标点不参与计分
#但要替换为空格，防止相邻英文词意外合并
ORDINARY_PUNCTUATION = "，。！？；：、,!?;:"


#mixed_tokens 的匹配优先级：
# 1. 英文词和技术表达
# 2. 数字串或小数
# 3. 中文字符
# 4. 其他非空白符号（fallback）
_TOKEN_PATTERN = re.compile(
    r"[a-z]+(?:[a-z0-9]|-[a-z0-9]+|\.\d+|'[a-z]+)*(?:\+\+|#)?"
    r"|\d+(?:\.\d+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002EBEF]"
    r"|[^\s]"
)

def normalize_score(text: str) -> str:
    """
    将原始文本转换为统一评分文本。

    规则：
    1. NFKC
    2. 英文小写
    3. 普通标点转为空格
    4. 非数字间的句点转为空格
    5. 折叠空白
    6. 保留未知有效符号
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    #1. Unicode兼容规范化
    text = unicodedata.normalize("NFKC", text)

    #2.英文小写(lower也会处理其他有大小写的文字)
    text = text.lower()

    #对不可安全解释的控制字符显式报错
    for char in text:
        category = unicodedata.category(char)

        if (
            category in {"Cc", "Cf", "Cs"}
            and not char.isspace()
        ):
            raise ValueError(
                f"Unsupported control character: U+{ord(char):04X}"
            )

    #3.普通标点替换为空格
    for punctuation in ORDINARY_PUNCTUATION:
        text = text.replace(punctuation, " ")

    #4. 数字之间的'.'保留：
    #    3.14、v1.2.3 保留
    #    hello.world -> hello world
    text = re.sub(
        r"(?<!\d)\.|\.(?!\d)",
        " ",
        text
    )

    #5.折叠空白
    text = " ".join(text.split())

    return text

def mixed_tokens(text: str) -> list[str]:
    """
    按项目 MER 规则生成混合 token。

    中文：逐字
    英文：按词
    数字：按连续数字串
    其他符号：保留为 fallback token
    """

    #统一评分入口
    normalized = normalize_score(text)

    #正则扫描
    tokens = _TOKEN_PATTERN.findall(normalized)

    return tokens