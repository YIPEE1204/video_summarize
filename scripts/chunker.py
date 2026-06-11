"""
文本分块模块（V0.3 新增）

用于超长视频文本的分段处理：
    1. 使用 tiktoken 精确统计 token 数
    2. 按时间轴将长文本切分为 token 数可控的块
    3. 每块保留完整的时间段信息，确保后续可分段总结

设计原则：
    - 仅在 snippet（时间戳）边界切分，不截断句子
    - 每块大小可配置，默认 40k tokens，适配 Claude 上下文窗口
    - 兼容 fetch_transcript() 和 ASR 两种数据源
"""

import tiktoken


# 默认最大 token 数（每块）
# Claude 上下文 200k，留出 160k 给其他对话内容
_DEFAULT_MAX_TOKENS = 40000

# 使用的 tokenizer 模型
# cl100k_base ≈ Claude tokenizer，用于粗略估算
_ENCODING = "cl100k_base"


def _get_encoder():
    """获取 tiktoken 编码器（带缓存）"""
    return tiktoken.get_encoding(_ENCODING)


def count_tokens(text: str) -> int:
    """
    统计文本的 token 数量。

    Args:
        text: 输入文本

    Returns:
        int: token 数量
    """
    encoder = _get_encoder()
    return len(encoder.encode(text))


def count_snippets_tokens(snippets: list) -> int:
    """
    统计一组 snippet 的总 token 数。

    Args:
        snippets: [{"text": str, ...}, ...]

    Returns:
        int: token 数量
    """
    encoder = _get_encoder()
    total = 0
    for sn in snippets:
        total += len(encoder.encode(sn.get("text", "")))
    return total


def chunk_snippets(
    snippets: list,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> list:
    """
    将字幕/ASR snippets 按 token 数切分为多个块。

    切分规则：
        - 严格在 snippet 边界处切分
        - 单个 snippet 超过 max_tokens 时独立成一块
        - 每块尽量接近但不超过 max_tokens

    Args:
        snippets: [
            {"start": float, "duration": float, "text": str},
            ...
        ]
        max_tokens: 每块最大 token 数（默认 40000）

    Returns:
        list[dict]: [
            {
                "chunk_index": int,       # 块序号（从 1 开始）
                "total_chunks": int,      # 总块数
                "start_time": float,      # 该块起始时间
                "end_time": float,        # 该块结束时间
                "snippets": list,         # 该块包含的 snippets
                "token_count": int,       # 该块的 token 数
                "full_text": str,         # 该块的纯文本
            },
            ...
        ]

    Raises:
        ValueError: snippets 为空
    """
    if not snippets:
        raise ValueError("snippets 为空，无法分块")

    encoder = _get_encoder()
    chunks = []
    current_snippets = []
    current_tokens = 0

    for sn in snippets:
        sn_text = sn.get("text", "")
        sn_tokens = len(encoder.encode(sn_text))

        # 单个 snippet 超过 max_tokens → 独立成块
        if sn_tokens > max_tokens:
            # 先把当前累积的块保存
            if current_snippets:
                chunks.append(_build_chunk(current_snippets, encoder))
                current_snippets = []
                current_tokens = 0
            # 这个超大的 snippet 单独成块
            chunks.append(_build_chunk([sn], encoder))
            continue

        # 如果加上这个 snippet 会超限 → 先保存当前块，再开始新块
        if current_tokens + sn_tokens > max_tokens and current_snippets:
            chunks.append(_build_chunk(current_snippets, encoder))
            current_snippets = []
            current_tokens = 0

        current_snippets.append(sn)
        current_tokens += sn_tokens

    # 最后一块
    if current_snippets:
        chunks.append(_build_chunk(current_snippets, encoder))

    # 标记序号和总分块数
    total = len(chunks)
    for i, ch in enumerate(chunks):
        ch["chunk_index"] = i + 1
        ch["total_chunks"] = total

    return chunks


def _build_chunk(snippets: list, encoder) -> dict:
    """根据一组 snippet 构建一个 chunk 字典"""
    full_text_parts = []
    for sn in snippets:
        full_text_parts.append(sn.get("text", ""))

    full_text = "\n".join(full_text_parts)

    return {
        "chunk_index": 0,  # 由调用者填充
        "total_chunks": 0,
        "start_time": snippets[0].get("start", 0),
        "end_time": snippets[-1].get("start", 0) + snippets[-1].get("duration", 0),
        "snippets": snippets,
        "token_count": len(encoder.encode(full_text)),
        "full_text": full_text,
    }


def should_chunk(snippets: list, max_tokens: int = _DEFAULT_MAX_TOKENS) -> bool:
    """
    判断是否需要分块处理。

    Args:
        snippets: 字幕/ASR snippets
        max_tokens: 阈值

    Returns:
        bool: True 表示需要分块
    """
    total = count_snippets_tokens(snippets)
    return total > max_tokens
