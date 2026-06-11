#!/usr/bin/env python3
"""
YouTube 视频总结 Skill — 主入口

Claude Code Skill：接收视频 URL，提取字幕内容，辅助生成结构化总结。

调用方式：
    python skill.py <youtube_url>
    python skill.py <youtube_url> -o output.txt
    python skill.py <youtube_url> --asr                 # 强制语音识别
    python skill.py <youtube_url> --no-asr              # 禁止 ASR 降级
    python skill.py <youtube_url> --max-tokens 30000    # 自定义分块阈值
"""

import argparse
import sys
import os
import json

# 将项目根目录和 scripts 目录加入 sys.path
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPTS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# 尝试加载配置
_CONFIG = {}
_config_path = os.path.join(_ROOT_DIR, "config.yaml")
try:
    import yaml
    if os.path.exists(_config_path):
        with open(_config_path, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f) or {}
except ImportError:
    pass  # 无 PyYAML 时使用默认配置

from scripts import youtube
from scripts import transcriber
from scripts import chunker
from scripts import frameextractor


def _check_environment():
    """检查是否在 Conda 虚拟环境中，不在则给出提示"""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if not conda_prefix:
        print("⚠ 建议在 Conda 虚拟环境中运行", file=sys.stderr)
        print("  本工具依赖 conda 安装 ffmpeg，推荐创建独立环境：", file=sys.stderr)
        print(f"   conda create -n summarize_env python=3.10 -y", file=sys.stderr)
        print(f"   conda activate summarize_env", file=sys.stderr)
        print(f"   conda install ffmpeg -y", file=sys.stderr)
        print(f"   pip install -r requirements.txt", file=sys.stderr)
        print(file=sys.stderr)
    elif conda_env and conda_env != "base":
        pass  # 在专用虚拟环境中，正常
    else:
        print(f"⚠ 当前在 base 环境中运行，建议创建专用虚拟环境以避免包冲突", file=sys.stderr)
        print(file=sys.stderr)


def load_config(section: str, key: str, default=None):
    """从配置文件中读取值，不存在则返回默认值"""
    try:
        return _CONFIG[section][key]
    except (KeyError, TypeError):
        return default


def normalize_text_data(video_id: str, raw_data: dict,
                        source: str = "transcript") -> dict:
    """
    统一字幕数据与 ASR 数据的格式。

    Args:
        video_id: 视频 ID
        raw_data: fetch_transcript() 或 transcribe() 的输出
        source: "transcript" 或 "asr"

    Returns:
        标准化后的 dict（包含 language, language_code, is_generated, snippets）
    """
    if source == "transcript":
        return {**raw_data, "video_id": video_id}

    return {
        "video_id": video_id,
        "language": raw_data.get("language", "zh"),
        "language_code": raw_data.get("language_code", "zh"),
        "is_generated": True,
        "source": "asr",
        "model_size": raw_data.get("model_size", "base"),
        "transcription_time": raw_data.get("transcription_time", 0),
        "snippets": raw_data.get("snippets", []),
        "full_text": raw_data.get("full_text", ""),
    }


def _calc_total_duration(snippets: list) -> float:
    """从 snippets 计算总时长"""
    if not snippets:
        return 0
    last = snippets[-1]
    return last.get("start", 0) + last.get("duration", 0)


def _get_source_label(data: dict) -> str:
    """获取数据来源的中文标签"""
    return "语音识别（ASR）" if data.get("source") == "asr" else "字幕"


def _build_info_lines(video_id: str, data: dict, file_path: str,
                      is_chunked: bool = False, saved: bool = True) -> list:
    """构建输出头部信息行（视频信息、元数据），两种模式共用"""
    total_seconds = _calc_total_duration(data["snippets"])
    is_asr = data.get("source") == "asr"
    source_label = _get_source_label(data)

    mode = "（分段模式）" if is_chunked else ""
    lines = [
        "=" * 60,
        f"  YouTube 视频内容总结{mode}",
        "=" * 60,
        "",
        f"视频链接：https://www.youtube.com/watch?v={video_id}",
        f"{source_label}语言：{data['language']} ({data['language_code']})",
        f"文本来源：{source_label}",
    ]

    if is_asr:
        lines.append(f"ASR 模型：{data.get('model_size', 'base')}")
        trans_time = data.get("transcription_time", 0)
        lines.append(f"转写耗时：{trans_time:.1f} 秒")

    lines.extend([
        f"{source_label}条数：{len(data['snippets'])} 条",
        f"音频时长：约 {_fmt_duration(total_seconds)}",
        "",
    ])
    if saved:
        lines.append(f"文本已保存至：{file_path}")
    else:
        lines.append(f"文本路径：{file_path}（未保存到磁盘）")
    lines.append("")
    return lines


def _build_summary_instructions(is_chunked: bool = False) -> list:
    """构建总结指令（单块 / 多块模式）"""
    if not is_chunked:
        return [
            "─" * 60,
            "  总结说明",
            "─" * 60,
            "",
            "请根据以上内容，生成一份结构化的视频总结，包含以下部分：",
            "",
            "1. 【视频信息】标题、语言、时长",
            "2. 【核心观点】视频的主要论点/知识点（分条列举）",
            "3. 【详细时间轴】按段落标注时间段和内容要点",
            "4. 【关键结论】视频最终的结论或建议",
            "",
            "要求：",
            "- 总结应完整还原视频全部信息要点，不追求简短",
            "- 对技术类内容保持术语准确",
            "- 如果内容有分章节/段落，按顺序归纳",
        ]

    return [
        "─" * 60,
        "  总结说明（分段模式）",
        "─" * 60,
        "",
        "由于视频较长，以上内容已按时间轴分为多段。请按以下步骤总结：",
        "",
        "步骤 1：逐段阅读上面每段内容，为每段生成核心要点。",
        "步骤 2：将所有段的要点合并，生成一份完整的结构化总结。",
        "",
        "最终输出应包含：",
        "1. 【视频信息】标题、语言、总时长",
        "2. 【核心观点】视频的主要论点/知识点（分条列举，跨段合并同类观点）",
        "3. 【详细时间轴】标注每个段落的起始时间、核心内容和关键时间节点",
        "4. 【关键结论】视频最终的结论或建议",
        "",
        "要求：",
        "- 总结应覆盖所有段落的内容，确保信息完整",
        "- 对技术类内容保持术语准确",
        "- 跨段重复的话题合并归纳，不要冗余",
    ]


def format_summary_prompt(video_id: str, data: dict,
                          transcript_file: str, saved: bool = True) -> str:
    """
    标准模式（单块）：输出完整的视频信息 + 全文 + 总结指令。
    """
    lines = _build_info_lines(video_id, data, transcript_file, saved=saved)

    is_asr = data.get("source") == "asr"
    lines.extend([
        "─" * 60,
        f"  视频{'字幕' if not is_asr else '转写文本'}全文（含时间戳）",
        "─" * 60,
        "",
    ])

    for snippet in data["snippets"]:
        start = snippet["start"]
        end = start + snippet.get("duration", 0)
        lines.append(f"[{_fmt_ts(start)} -> {_fmt_ts(end)}] {snippet['text']}")

    lines.extend(_build_summary_instructions(is_chunked=False))
    return "\n".join(lines)


def format_chunked_summary_prompt(video_id: str, data: dict,
                                  transcript_file: str,
                                  chunks: list, saved: bool = True) -> str:
    """
    分段模式（多块）：输出视频信息 + 逐段内容 + 分段总结指令。

    Args:
        video_id: 视频 ID
        data: 标准化后的文本数据
        transcript_file: 保存的文件路径
        chunks: chunker.chunk_snippets() 的输出，分块列表
    """
    lines = _build_info_lines(video_id, data, transcript_file, is_chunked=True, saved=saved)

    is_asr = data.get("source") == "asr"
    source_label = "字幕" if not is_asr else "转写文本"

    # 输出分段概要
    lines.append(f"⚠️  内容较长，已自动分为 {len(chunks)} 段（每段约 "
                 f"{load_config('chunker', 'max_tokens_per_chunk', 40000)} tokens）")
    lines.append("")

    # 逐段输出
    for ch in chunks:
        lines.extend([
            "=" * 60,
            f"  {source_label} — 第 {ch['chunk_index']} 段 / 共 {ch['total_chunks']} 段"
            f"  [{_fmt_ts(ch['start_time'])} → {_fmt_ts(ch['end_time'])}]"
            f"  (约 {ch['token_count']} tokens)",
            "=" * 60,
            "",
        ])

        for snippet in ch["snippets"]:
            start = snippet["start"]
            end = start + snippet.get("duration", 0)
            lines.append(f"[{_fmt_ts(start)} -> {_fmt_ts(end)}] {snippet['text']}")

        lines.append("")

    lines.extend(_build_summary_instructions(is_chunked=True))
    return "\n".join(lines)


def save_text_to_file(data: dict, output_path: str,
                      source: str = "transcript",
                      chunks: list = None) -> str:
    """
    将文本数据保存为文件，支持分段模式分别保存。

    Args:
        data: 标准化后的文本数据
        output_path: 输出路径
        source: "transcript" 或 "asr"
        chunks: 分段列表（可选），有则逐段保存为独立文件

    Returns:
        str: 保存的文件路径（分段模式下为主文件路径）
    """
    is_asr = source == "asr"
    source_label = "语音识别" if is_asr else "字幕"

    if chunks:
        # 分段模式：保存主文件 + 每段独立文件
        return _save_chunked_files(data, output_path, source_label)
    else:
        # 单块模式：原逻辑
        return _save_single_file(data, output_path, source_label)


def _save_single_file(data: dict, output_path: str,
                      source_label: str) -> str:
    """保存单块文本文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"视频: https://www.youtube.com/watch?v={data['video_id']}\n")
        f.write(f"{source_label}语言: {data['language']} ({data['language_code']})\n")
        if data.get("source") == "asr":
            f.write(f"ASR 模型: {data.get('model_size', 'base')}\n")
        else:
            f.write(f"字幕类型: {'自动生成' if data.get('is_generated') else '手动上传'}\n")
        f.write(f"{source_label}条数: {len(data['snippets'])}\n")
        f.write("=" * 60 + "\n\n")

        for snippet in data["snippets"]:
            start = snippet["start"]
            end = start + snippet.get("duration", 0)
            f.write(f"[{_fmt_ts(start)} -> {_fmt_ts(end)}] {snippet['text']}\n")

    print(f"   ✓ 已保存到: {output_path}", file=sys.stderr)
    return output_path


def _save_chunked_files(data: dict, output_path: str,
                        source_label: str) -> str:
    """
    分段模式：保存主汇总文件 + 每段独立文件。

    主文件仅包含摘要元信息，详细内容按段拆分存储。
    """
    from scripts import chunker as chk_mod

    base, ext = os.path.splitext(output_path)
    chunks = chk_mod.chunk_snippets(
        data["snippets"],
        max_tokens=load_config("chunker", "max_tokens_per_chunk", 40000),
    )

    # 保存每段独立文件
    chunk_files = []
    for ch in chunks:
        chunk_path = f"{base}_part{ch['chunk_index']}{ext}"
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(f"视频: https://www.youtube.com/watch?v={data['video_id']}\n")
            f.write(f"{source_label}语言: {data['language']} ({data['language_code']})\n")
            f.write(f"分段: 第 {ch['chunk_index']}/{ch['total_chunks']} 段 "
                    f"[{_fmt_ts(ch['start_time'])} → {_fmt_ts(ch['end_time'])}]\n")
            f.write(f"token 数: 约 {ch['token_count']}\n")
            f.write("=" * 60 + "\n\n")

            for snippet in ch["snippets"]:
                start = snippet["start"]
                end = start + snippet.get("duration", 0)
                f.write(f"[{_fmt_ts(start)} -> {_fmt_ts(end)}] {snippet['text']}\n")

        chunk_files.append(chunk_path)

    # 保存主文件（汇总信息 + 分段索引）
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"视频: https://www.youtube.com/watch?v={data['video_id']}\n")
        f.write(f"{source_label}语言: {data['language']} ({data['language_code']})\n")
        f.write(f"{source_label}条数: {len(data['snippets'])} 条\n")
        f.write(f"分段数: {len(chunks)}\n")
        f.write("=" * 60 + "\n\n")

        for ch in chunks:
            f.write(f"--- 第 {ch['chunk_index']}/{ch['total_chunks']} 段 "
                    f"[{_fmt_ts(ch['start_time'])} → {_fmt_ts(ch['end_time'])}] "
                    f"(约 {ch['token_count']} tokens) ---\n")
            f.write(f"   独立文件: {os.path.basename(chunk_files[ch['chunk_index'] - 1])}\n\n")

    print(f"   ✓ 主文件: {output_path}", file=sys.stderr)
    for cf in chunk_files:
        print(f"   ✓ 分段文件: {cf}", file=sys.stderr)

    return output_path


def _build_visual_section(video_id: str, frames: list,
                           gemini_analysis: str = None) -> str:
    """
    构建视觉增强部分的内容（插入摘要 prompt 中）。

    Args:
        video_id: 视频 ID
        frames: extract_frames() 返回的帧列表
        gemini_analysis: Gemini 多模态分析结果（可选）

    Returns:
        str: 格式化后的视觉增强文本
    """
    lines = [
        "",
        "=" * 60,
        "  视觉增强 — 关键帧分析",
        "=" * 60,
        "",
        f"提取到 {len(frames)} 个关键帧，时间分布：",
    ]
    for f in frames:
        ts = f.get("timestamp", 0)
        lines.append(f"  - [{_fmt_ts(ts)}] {os.path.basename(f['path'])}")
    lines.append("")

    if gemini_analysis:
        lines.extend([
            "─" * 60,
            "  Gemini 多模态画面分析",
            "─" * 60,
            "",
            gemini_analysis,
            "",
        ])
    else:
        lines.extend([
            "⚠️  未配置 Gemini API Key，仅提取了关键帧图片。",
            "   如需画面内容分析，请配置 GEMINI_API_KEY。",
            "   帧文件保存在临时目录，如需长期保存请指定 --output-dir。",
            "",
        ])
    return "\n".join(lines)


def _run_visual_pipeline(video_id: str, args,
                          gemini_key: str = None) -> tuple:
    """
    执行视觉增强流程：提取关键帧 → 可选 Gemini 分析。

    Args:
        video_id: 视频 ID
        args: 命令行参数
        gemini_key: Gemini API Key（可选）

    Returns:
        tuple: (frames, gemini_analysis_or_None)
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    frames = []
    gemini_analysis = None

    print(f"🖼️  正在提取关键帧...", file=sys.stderr)
    sys.stderr.flush()

    try:
        frames = frameextractor.extract_frames(
            video_url,
            max_frames=load_config("gemini", "max_frames", 10),
        )
        print(f"  ✓ 完成！{len(frames)} 个关键帧", file=sys.stderr)

        # 如果有 API Key，进行多模态分析
        if gemini_key or load_config("gemini", "api_key", ""):
            api_key = gemini_key or load_config("gemini", "api_key", "")
            if not api_key:
                api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                model = load_config("gemini", "model", "gemini-1.5-pro")
                print(f"  📡 正在分析画面内容（模型: {model}）...",
                      file=sys.stderr)
                sys.stderr.flush()
                gemini_analysis = frameextractor.describe_frames_with_gemini(
                    frames, api_key=api_key, model_name=model,
                )
                print(f"  ✓ 画面分析完成", file=sys.stderr)
        else:
            print(f"  ⚠️  未配置 Gemini API Key，跳过画面分析",
                  file=sys.stderr)

    except (ImportError, RuntimeError, ValueError) as e:
        print(f"  ⚠️  视觉增强跳过: {e}", file=sys.stderr)

    return frames, gemini_analysis


def _fmt_ts(seconds: float) -> str:
    """将秒数格式化为 MM:SS 或 H:MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _fmt_duration(seconds: float) -> str:
    """格式化时长为可读字符串"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    parts = []
    if h > 0:
        parts.append(f"{h} 小时")
    if m > 0:
        parts.append(f"{m} 分钟")
    if s > 0 or not parts:
        parts.append(f"{s} 秒")
    return "".join(parts)


def _run_asr_pipeline(video_id: str, args) -> dict:
    """
    执行 ASR 完整流程：获取视频信息 → 下载音频 → 语音转写 → 清理。

    Returns:
        dict: 标准化后的文本数据
    """
    print(f"ℹ️  视频无可用字幕，启动语音识别 (ASR)...", file=sys.stderr)

    asr_model = load_config("whisper", "model_size", "base")
    asr_device = load_config("whisper", "device", "cpu")
    asr_language = load_config("whisper", "language", None)

    print(f"⬇️  正在下载音频 (模型: {asr_model}, 设备: {asr_device})...",
          file=sys.stderr)
    sys.stderr.flush()

    audio_info = youtube.download_audio(video_id)
    audio_path = audio_info["file_path"]
    duration = audio_info["duration"]
    title = audio_info["title"]

    print(f"   ✓ 标题: {title}", file=sys.stderr)
    print(f"   ✓ 时长: {_fmt_duration(duration)}", file=sys.stderr)
    print(f"   ✓ 音频: {audio_path}", file=sys.stderr)
    sys.stderr.flush()

    try:
        asr_result = transcriber.transcribe(
            audio_path=audio_path,
            model_size=asr_model,
            device=asr_device,
            language=asr_language,
            verbose=True,
        )
    except Exception as e:
        youtube.cleanup_audio(audio_path)
        raise

    youtube.cleanup_audio(audio_path)
    return normalize_text_data(video_id, asr_result, source="asr")


def main():
    # 检查运行环境（Conda）
    _check_environment()

    # 确保 stdout 使用 UTF-8（避免 Windows GBK 终端报错）
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="YouTube 视频总结工具 — 提取字幕 / 语音转写并生成结构化内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python skill.py https://www.youtube.com/watch?v=abc123\n"
            "  python skill.py https://youtu.be/abc123 -o output.txt\n"
            "  python skill.py https://youtu.be/abc123 --asr\n"
            "  python skill.py https://youtu.be/abc123 --max-tokens 30000\n"
        ),
    )
    parser.add_argument("url", help="YouTube 视频 URL 或视频 ID")
    parser.add_argument("-o", "--output",
                        help="输出文件路径（默认: {video_id}_transcript.txt）")
    parser.add_argument("--lang", nargs="+", default=None,
                        help="字幕语言优先级（默认: zh-Hans zh en）")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 格式输出结构化数据（调试用）")
    parser.add_argument("--asr", action="store_true",
                        help="强制使用语音识别（即使有字幕）")
    parser.add_argument("--no-asr", action="store_true",
                        help="禁止无字幕时降级到语音识别")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="分块阈值 token 数（默认 40000）")
    parser.add_argument("--visual", action="store_true",
                        help="启用视觉增强：提取关键帧（需下载视频）")
    parser.add_argument("--gemini-key", type=str, default=None,
                        help="Gemini API Key（视觉增强模式可选，否则从配置或环境变量读取）")
    parser.add_argument("--keep", action="store_true",
                        help="长期保存字幕/转写文本到磁盘（默认仅输出到 stdout，不写文件）")
    args = parser.parse_args()

    # 加载配置覆盖
    default_langs = load_config("youtube", "default_languages", None)
    if args.lang is None and default_langs:
        args.lang = default_langs

    max_tokens = args.max_tokens or load_config("chunker", "max_tokens_per_chunk", 40000)

    # ========== 步骤 1：提取视频 ID ==========
    print(f"🎬 正在解析视频链接...", file=sys.stderr)
    try:
        video_id = youtube.extract_video_id(args.url)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    print(f"   ✓ 视频 ID: {video_id}", file=sys.stderr)

    # ========== 步骤 2：获取文本内容 ==========
    data = None
    source = "transcript"

    if not args.asr:
        print(f"📝 正在获取字幕...", file=sys.stderr)
        try:
            data = youtube.fetch_transcript(video_id, languages=args.lang)
            print(f"   ✓ 语言: {data['language']} ({data['language_code']})",
                  file=sys.stderr)
            print(f"   ✓ 共 {len(data['snippets'])} 条字幕", file=sys.stderr)
            print(f"   ✓ 类型: {'自动生成' if data['is_generated'] else '手动上传'}",
                  file=sys.stderr)
        except youtube.TranscriptsDisabled:
            if args.no_asr:
                print(f"❌ 该视频没有字幕，且已指定 --no-asr", file=sys.stderr)
                sys.exit(1)
            print(f"   ⚠️  无可用字幕", file=sys.stderr)
        except youtube.NoTranscriptFound:
            if args.no_asr:
                print(f"❌ 找不到匹配字幕，且已指定 --no-asr", file=sys.stderr)
                sys.exit(1)
            print(f"   ⚠️  无匹配语言字幕", file=sys.stderr)
        except youtube.VideoUnavailable:
            print(f"❌ 视频不可用（可能已删除或设为私密）", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"⚠️  获取字幕异常: {e}", file=sys.stderr)
    else:
        print(f"   ⚙️  已指定 --asr，跳过字幕直接使用语音识别", file=sys.stderr)

    # 字幕获取失败 → ASR 降级
    if data is None:
        source = "asr"
        try:
            data = _run_asr_pipeline(video_id, args)
        except ImportError as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as e:
            print(f"❌ ASR 流程失败: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ ASR 流程异常: {e}", file=sys.stderr)
            sys.exit(1)

    # ========== 步骤 3：检测是否需要分段 ==========
    total_tokens = chunker.count_snippets_tokens(data["snippets"])
    needs_chunk = total_tokens > max_tokens

    if needs_chunk:
        print(f"📐 内容较长 ({total_tokens:,} tokens，阈值 {max_tokens:,})，"
              f"自动分段处理...", file=sys.stderr)
        chunks = chunker.chunk_snippets(data["snippets"], max_tokens=max_tokens)
        print(f"   ✓ 已分为 {len(chunks)} 段", file=sys.stderr)
    else:
        chunks = None
        print(f"   ✓ Token 数: {total_tokens:,}（无需分段）", file=sys.stderr)

    # ========== 步骤 4：保存文件（可选） ==========
    output_dir = load_config("output", "directory", "")
    if args.output:
        output_path = args.output
    elif output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}_transcript.txt")
    else:
        output_path = f"{video_id}_transcript.txt"

    do_save = args.keep or args.output is not None
    if do_save:
        try:
            save_text_to_file(data, output_path, source=source, chunks=chunks)
        except Exception as e:
            print(f"⚠️  保存文件失败: {e}", file=sys.stderr)
    else:
        print(f"   ℹ️  文本内容已生成，未保存到磁盘", file=sys.stderr)
        print(f"   ℹ️  如需长期保存，请添加 --keep 参数", file=sys.stderr)

    # ========== 步骤 4.5：视觉增强（可选） ==========
    frames = []
    gemini_analysis = None
    if args.visual:
        visual_key = args.gemini_key
        if not visual_key:
            visual_key = load_config("gemini", "api_key", None)
        frames, gemini_analysis = _run_visual_pipeline(
            video_id, args, gemini_key=visual_key,
        )

    # ========== 步骤 5：输出 ==========
    if args.json:
        output = json.dumps(data, ensure_ascii=False, indent=2)
        if needs_chunk:
            # JSON 模式附带分块信息
            chunk_info = json.dumps(
                [{"index": c["chunk_index"], "total": c["total_chunks"],
                  "start": _fmt_ts(c["start_time"]), "end": _fmt_ts(c["end_time"]),
                  "tokens": c["token_count"]}
                 for c in chunks],
                ensure_ascii=False, indent=2,
            )
            output = json.dumps({
                "data": data,
                "chunks": chunks,
                "total_tokens": total_tokens,
            }, ensure_ascii=False, indent=2)
        print(output)
    else:
        if needs_chunk:
            output = format_chunked_summary_prompt(
                video_id, data, output_path, chunks, saved=do_save,
            )
        else:
            output = format_summary_prompt(
                video_id, data, output_path, saved=do_save,
            )

        # 追加视觉增强部分
        if frames:
            output += _build_visual_section(
                video_id, frames, gemini_analysis=gemini_analysis,
            )

        print(output)

    # 最终状态报告
    if do_save:
        print(f"\n✨ 完成！文本文件: {output_path}", file=sys.stderr)
    else:
        print(f"\n✨ 完成！文本已输出到 stdout", file=sys.stderr)


if __name__ == "__main__":
    main()
