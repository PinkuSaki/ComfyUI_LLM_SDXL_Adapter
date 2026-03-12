"""
ComfyUI 风格的 prompt 权重解析器。

支持语法：
- (tag)        隐式权重 1.1
- (tag:1.5)    显式权重 1.5
- ((tag))      嵌套隐式 1.1^2 = 1.21
- ((tag:1.5))  显式权重 1.5（不继续乘外层隐式权重）
- \\( \\)      转义为字面括号
"""

from __future__ import annotations


def _find_matching_paren(text, start_index):
    """找到与 text[start_index] 对应的右括号位置。"""
    depth = 0
    i = start_index

    while i < len(text):
        ch = text[i]

        if ch == "\\" and i + 1 < len(text) and text[i + 1] in "()":
            i += 2
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return None


def _split_explicit_weight(text):
    """
    如果文本尾部存在顶层的 :weight 语法，返回 (content, weight)。
    否则返回 (原文本, None)。
    """
    depth = 0
    last_top_level_colon = -1
    i = 0

    while i < len(text):
        ch = text[i]

        if ch == "\\" and i + 1 < len(text) and text[i + 1] in "()":
            i += 2
            continue

        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        elif ch == ":" and depth == 0:
            last_top_level_colon = i

        i += 1

    if last_top_level_colon == -1:
        return text, None

    weight_text = text[last_top_level_colon + 1 :].strip()
    if not weight_text:
        return text, None

    try:
        weight = float(weight_text)
    except ValueError:
        return text, None

    return text[:last_top_level_colon], weight


def _merge_segments(segments):
    if not segments:
        return [("", 1.0)]

    merged = [segments[0]]
    for span_text, weight in segments[1:]:
        if abs(weight - merged[-1][1]) < 1e-6:
            merged[-1] = (merged[-1][0] + span_text, weight)
        else:
            merged.append((span_text, weight))

    return merged


def _parse_segments(text, base_weight):
    segments = []
    literal_buffer = []
    i = 0

    while i < len(text):
        ch = text[i]

        if ch == "\\" and i + 1 < len(text) and text[i + 1] in "()":
            literal_buffer.append(text[i + 1])
            i += 2
            continue

        if ch == "(":
            closing_index = _find_matching_paren(text, i)
            if closing_index is None:
                literal_buffer.append(ch)
                i += 1
                continue

            if literal_buffer:
                segments.append(("".join(literal_buffer), base_weight))
                literal_buffer = []

            inner_text = text[i + 1 : closing_index]
            explicit_content, explicit_weight = _split_explicit_weight(inner_text)
            group_weight = explicit_weight if explicit_weight is not None else base_weight * 1.1

            segments.extend(_parse_segments(explicit_content, group_weight))
            i = closing_index + 1
            continue

        literal_buffer.append(ch)
        i += 1

    if literal_buffer:
        segments.append(("".join(literal_buffer), base_weight))

    return _merge_segments(segments)


def parse_weighted_prompt(text):
    """
    解析带权重标记的 prompt 文本。

    返回 [(text_span, weight), ...] 列表。所有 span 拼接后等于
    去除权重语法后的原始文本；转义括号会还原为字面字符。

    >>> parse_weighted_prompt("a, (b:1.5), c")
    [('a, ', 1.0), ('b', 1.5), (', c', 1.0)]
    >>> parse_weighted_prompt("((tag))")
    [('tag', 1.2100000000000002)]
    >>> parse_weighted_prompt("((tag:1.5))")
    [('tag', 1.5)]
    >>> parse_weighted_prompt("(foo (bar):1.5)")
    [('foo ', 1.5), ('bar', 1.6500000000000001)]
    >>> parse_weighted_prompt("\\\\(literal\\\\)")
    [('(literal)', 1.0)]
    """
    return _merge_segments(_parse_segments(text, 1.0))


def build_weighted_character_map(text):
    """
    返回去除权重语法后的纯文本，以及与字符一一对应的权重列表。
    """
    segments = parse_weighted_prompt(text)
    plain_text = "".join(span_text for span_text, _ in segments)
    char_weights = []

    for span_text, weight in segments:
        char_weights.extend([weight] * len(span_text))

    return plain_text, char_weights
