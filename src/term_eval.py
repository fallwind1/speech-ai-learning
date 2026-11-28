
import csv
import json

from pathlib import Path
from collections import defaultdict

from src.normalize import mixed_tokens
from src.normalize import mixed_tokens
from src.metrics import score_tokens


def load_terms(path):
    """
    加载术语词表。

    输入：
        terms.csv 路径

    输出：
        {
            "entries": [...],
            "matches_by_first_token": {...}
        }
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    entries = []

    #检查 canonical_id是否重复
    seen_ids = set()

    #检查不同ID是否共享相同别名
    alias_owner = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.DictReader(f)

        required_columns = {
            "canonical_id",
            "canonical",
            "aliases"
        }

        if reader.fieldnames is None:
            raise ValueError("CSV 文件缺少表头")

        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV 存在重复列名")

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"CSV 缺少必要字段: {missing}"
            )

        for line_no, row in enumerate(reader, start=2):
            # 跳过完全空白的行
            if all(
                value is None or not str(value).strip()
                for value in row.values()
            ):
                continue

            if None in row:
                raise ValueError(
                    f"第 {line_no} 行存在多余字段"
                )

            # 1. 读取基本字段
            canonical_id = (
                row["canonical_id"] or ""
            ).strip()

            canonical = (
                row["canonical"] or ""
            ).strip()

            aliases_raw = row["aliases"]

            if not canonical_id:
                raise ValueError(
                    f"第 {line_no} 行 canonical_id 为空"
                )

            if not canonical:
                raise ValueError(
                    f"第 {line_no} 行 canonical 为空"
                )

             # 2. 检查 ID 唯一性
            if canonical_id in seen_ids:
                raise ValueError(
                    f"重复 canonical_id: {canonical_id}"
                )

            seen_ids.add(canonical_id)

            # 3. 解析 JSON aliases
            try:
                aliases = json.loads(aliases_raw)

            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"第 {line_no} 行 aliases 不是合法 JSON"
                ) from exc

            if not isinstance(aliases, list):
                raise ValueError(
                    f"第 {line_no} 行 aliases 必须是列表"
                )

            # 4. canonical 和 aliases 都需要参与匹配
            forms = [canonical] + aliases

            # 同一个 ID 内部避免重复保存相同形式
            seen_forms = set()

            for form in forms:

                if not isinstance(form, str):
                    raise ValueError(
                        f"第 {line_no} 行存在非字符串别名"
                    )

                form = form.strip()

                if not form:
                    raise ValueError(
                        f"第 {line_no} 行存在空别名"
                    )

                 # 5. 转换为 token 序列
                token_seq = tuple(mixed_tokens(form))

                if not token_seq:
                    raise ValueError(
                        f"第 {line_no} 行别名无法生成有效 token"
                    )

                # 同一术语内的重复形式不需要重复保存
                if token_seq in seen_forms:
                    continue

                seen_forms.add(token_seq)

                # 6. 检查不同 ID 之间是否存在别名冲突
                owner = alias_owner.get(token_seq)

                if (
                    owner is not None
                    and owner != canonical_id
                ):
                    raise ValueError(
                        f"别名冲突: {form!r} "
                        f"同时映射到 {owner} 和 {canonical_id}"
                    )

                alias_owner[token_seq] = canonical_id

                # 7. 保存匹配记录
                entries.append({
                    "canonical_id": canonical_id,
                    "canonical": canonical,
                    "form": form,
                    "tokens": token_seq,
                    "order": len(entries)
                })

    if not entries:
        raise ValueError("术语词表为空")

    # 8. 按首个 token 建立索引
    matches_by_first_token = defaultdict(list)

    for entry in entries:
        first_token = entry["tokens"][0]

        matches_by_first_token[first_token].append(entry)

    # 9. 最长匹配优先；等长时保持词表顺序
    for candidates in matches_by_first_token.values():

        candidates.sort(
            key=lambda item: (
                -len(item["tokens"]),
                item["order"]
            )
        )

    return {
        "entries": entries,
        "matches_by_first_token": dict(
            matches_by_first_token
        )
    }


def atomicize(tokens, terms):
    """
    将完整术语转换为 TERM:id。

    规则：
    1. 从左到右扫描
    2. 最长匹配优先
    3. 等长按照词表顺序
    4. 普通 token 保留
    5. 重复术语保留
    """
    if isinstance(tokens, str) or not isinstance(tokens, (list, tuple)):
        raise TypeError("tokens 必须是 list 或 tuple")

    if not all(isinstance(token, str) for token in tokens):
        raise TypeError("tokens 中的元素必须是字符串")

    if (
        not isinstance(terms, dict)
        or "matches_by_first_token" not in terms
    ):
        raise ValueError("terms 不是有效的 load_terms() 返回结果")

    tokens = tuple(tokens)

    matches = terms["matches_by_first_token"]

    result = []

    i = 0

    while i < len(tokens):

        current_token = tokens[i]

        # 1. 根据首个 token 查找候选术语
        candidates = matches.get(
            current_token,
            []
        )

        matched_entry = None

        # 2. 候选已经按最长匹配排序
        for entry in candidates:

            alias_tokens = entry["tokens"]

            length = len(alias_tokens)

            #3. 比较当前位置开始的token序列
            if tokens[i:i + length] == alias_tokens:

                matched_entry = entry

                break

        #4. 如果找到匹配术语
        if matched_entry is not None:

            result.append(
                f"TERM:{matched_entry['canonical_id']}"
            )

            # 跳过已经匹配的 token
            i += len(matched_entry["tokens"])

        else:

            # 5. 没有匹配则保留普通 token
            result.append(current_token)

            i += 1

    return result

def is_term(token):
    return (
        isinstance(token, str)
        and token.startswith("TERM:")
    )

def align_sequences(reference_atoms, hypothesis_atoms):

    return score_tokens(
        reference_atoms,
        hypothesis_atoms
    )

def compute_term_stats(
    reference_atoms,
    operations
):
    """
    根据完整句子 alignment 统计：

    S_t: 术语替换
    D_t: 术语删除
    I_t: 术语插入
    N_t: Reference 术语总次数
    """

    counts = {
        "S_t": 0,
        "D_t": 0,
        "I_t": 0,
        "N_t": 0
    }

    counts["N_t"] = sum(
        is_term(token)
        for token in reference_atoms
    )

    for op, ref_token, hyp_token in operations:

        ref_is_term = is_term(ref_token)
        hyp_is_term = is_term(hyp_token)

        if op == "S":
            if ref_is_term is True:
                counts["S_t"] += 1
            if ref_is_term is False and hyp_is_term is True:
                counts["I_t"] += 1
        if op == "D":
            if ref_is_term is True:
                counts["D_t"] += 1
        if op == "I":
            if hyp_is_term is True:
                counts["I_t"] += 1

    if counts["N_t"] > 0 :
        term_wer = (
            counts["S_t"]
            + counts["D_t"]
            + counts["I_t"]
        ) / counts["N_t"]
    else:
        term_wer = None

    return {
        **counts,
        "term_wer": term_wer
    }


def score_term_corpus(token_pairs):
    """
    计算语料级 term-WER。

    token_pairs:
        [
            (reference_atoms, hypothesis_atoms),
            ...
        ]

    返回：
        语料级 S_t、D_t、I_t、N_t、
        term_errors、term_wer 和 sample_results。
    """

    totals = {
        "S_t": 0,
        "D_t": 0,
        "I_t": 0,
        "N_t": 0
    }

    sample_results = []

    for sample_index, pair in enumerate(token_pairs, start=1):

        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                "每个样本必须包含 Reference 和 Hypothesis"
            )

        reference_atoms, hypothesis_atoms = pair

        # TODO 1：获得完整 alignment
        alignment = align_sequences(reference_atoms,hypothesis_atoms)

        # TODO 2：计算当前音频的术语统计
        information =compute_term_stats(reference_atoms,alignment["operations"])

        # TODO 3：累计 S_t、D_t、I_t、N_t
        for key in totals:
            totals[key] += information[key]

        # TODO 4：保存当前音频的统计结果
        sample_results.append({
        "sample_index": sample_index,
        **information
    })

    # TODO 5：计算语料级术语错误总数
    term_errors = totals["D_t"] + totals["I_t"] + totals["S_t"]

    # TODO 6：计算语料级 term-WER
    # 注意 N_total = 0 的情况
    if totals["N_t"] > 0:
        term_wer = term_errors / totals["N_t"]
    else:
        term_wer = None

    return {
        **totals,
        "term_errors": term_errors,
        "term_wer": term_wer,
        "sample_results": sample_results
    }
