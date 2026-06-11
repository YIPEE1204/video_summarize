"""
视频关键帧提取模块（V0.4）

提供关键帧提取与 Gemini 多模态分析功能。

关键帧提取流程：
    1. yt-dlp 下载视频流（仅视频，720p，无音频）
    2. ffmpeg 提取 I 帧（skip_frame nokey，天然对应场景切换点）
    3. 感知哈希去重（移除画面相似的冗余帧）
    4. 均匀采样最多 max_frames 帧
    5. 清理临时视频文件

多模态分析流程（需 Gemini API Key）：
    1. 将关键帧 + 字幕文本发送到 Gemini 1.5 Pro
    2. 获取结构化视觉描述

使用示例：
    from scripts import frameextractor

    # 仅提取关键帧（无 Gemini）
    frames = frameextractor.extract_frames("https://youtu.be/abc123")
    for f in frames:
        print(f"  [{f['timestamp']:.1f}s] {f['path']}")

    # 带 Gemini 分析
    desc = frameextractor.describe_frames_with_gemini(frames, api_key="...")
"""

import os
import re
import sys
import tempfile
import subprocess
from PIL import Image


def _find_ffmpeg_dir() -> str:
    """复用 youtube 模块的 ffmpeg 路径查找逻辑"""
    from .youtube import _find_ffmpeg
    return _find_ffmpeg()


def _average_hash(image: Image.Image, hash_size: int = 8) -> int:
    """
    计算图片的平均感知哈希（Average Hash）。

    将图片缩放到 hash_size × hash_size，转为灰度，
    每位标记像素值是否高于平均值。
    """
    img = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    return sum((1 << i) for i, p in enumerate(pixels) if p > avg)


def _hamming_distance(h1: int, h2: int) -> int:
    """计算两个哈希值的汉明距离（不同的位数）"""
    return (h1 ^ h2).bit_count()


def _deduplicate_frames(frame_paths: list, threshold: int = 10) -> list:
    """
    使用感知哈希去重，移除画面相似的冗余帧。

    Args:
        frame_paths: 原始帧文件路径列表（已排序）
        threshold: 汉明距离阈值，越小越严格（建议 5-15）

    Returns:
        list[str]: 去重后的帧路径列表
    """
    unique = []
    hashes = []

    for path in frame_paths:
        try:
            img = Image.open(path)
            h = _average_hash(img)
            img.close()

            is_dup = any(_hamming_distance(h, seen) < threshold for seen in hashes)
            if not is_dup:
                unique.append(path)
                hashes.append(h)
            else:
                try:
                    os.remove(path)
                except Exception:
                    pass
        except Exception:
            unique.append(path)  # 出错时保留

    return unique


def _parse_pts_from_filename(filepath: str) -> float:
    """
    从 ffmpeg -frame_pts 1 输出的文件名解析 PTS。

    文件名格式如 keyframe_00123.png → PTS = 123
    PTS 单位为 stream time_base，常见 mp4 为毫秒（time_base=1/1000）。
    """
    basename = os.path.basename(filepath)
    match = re.search(r'(\d+)', basename)
    if match:
        return float(match.group(1))
    return 0.0


def _estimate_timestamps(frame_paths: list, video_duration: float) -> list:
    """
    估算每帧在视频中的时间位置。

    优先解析 ffmpeg PTS 值，若不合理则按均匀分布估算。
    """
    if not frame_paths:
        return []
    if video_duration <= 0:
        return [0.0] * len(frame_paths)

    # 尝试从文件名 PTS 解析
    pts_values = [_parse_pts_from_filename(p) for p in frame_paths]
    max_pts = max(pts_values) if pts_values else 0

    # 判断 PTS 是否合理：max_pts 应在 [1, video_duration * 2] 范围内
    if max_pts > 1.0 and max_pts <= video_duration * 2:
        return pts_values

    # 否则按均匀分布估算（PTS 不可靠时的兜底）
    if len(frame_paths) <= 1:
        return [0.0]
    interval = video_duration / len(frame_paths)
    return [i * interval for i in range(len(frame_paths))]


def _select_frames_evenly(frame_paths: list, video_duration: float,
                           max_frames: int) -> list:
    """
    从去重后的关键帧中均匀选出最多 max_frames 帧。

    Returns:
        list[dict]: [{"path": str, "timestamp": float}, ...]
    """
    if not frame_paths:
        return []

    timestamps = _estimate_timestamps(frame_paths, video_duration)
    zipped = list(zip(frame_paths, timestamps))

    if len(zipped) <= max_frames:
        return [{"path": p, "timestamp": t} for p, t in zipped]

    # 均匀采样（取每段中间位置）
    step = len(zipped) / max_frames
    selected = []
    for i in range(max_frames):
        idx = min(int(i * step + step / 2), len(zipped) - 1)
        p, t = zipped[idx]
        selected.append({"path": os.path.abspath(p), "timestamp": t})

    return selected


def extract_frames(video_url: str, max_frames: int = 10,
                   output_dir: str = None) -> list:
    """
    从视频中提取关键帧。

    完整流程：
        1. yt-dlp 获取视频信息（标题、时长）
        2. yt-dlp 下载视频流（720p，仅视频，无音频）
        3. ffmpeg -skip_frame nokey 提取 I 帧
        4. 感知哈希去重
        5. 均匀采样最多 max_frames 帧
        6. 清理临时视频文件

    Args:
        video_url: YouTube 视频 URL 或视频 ID
        max_frames: 最多保留帧数（默认 10）
        output_dir: 帧图片输出目录（默认系统临时目录）

    Returns:
        list[dict]: 按时间戳排序的帧列表，每项含 path 和 timestamp

    Raises:
        ImportError: yt-dlp 未安装
        RuntimeError: ffmpeg 未找到 / 下载失败 / 无帧提取
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp 未安装。请运行: pip install yt-dlp")

    # 查找 ffmpeg
    ffmpeg_dir = _find_ffmpeg_dir()
    if not ffmpeg_dir:
        raise RuntimeError(
            "未找到 ffmpeg，无法提取关键帧。\n"
            "请安装：conda install -n summarize_env ffmpeg"
        )
    ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")

    # 输出目录
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="ytframes_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    # === 1. 获取视频信息 ===
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                               "skip_download": True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            duration = info.get("duration", 0)
            title = info.get("title", video_url)
    except Exception as e:
        raise RuntimeError(f"获取视频信息失败: {e}")

    print(f"  ✓ 视频: {title}", file=sys.stderr)
    print(f"  ✓ 时长: {duration // 60}:{duration % 60:02d}", file=sys.stderr)

    # === 2. 下载视频流（仅视频，720p，无音频） ===
    video_path = os.path.join(output_dir, "video.mp4")
    ydl_opts = {
        "format": "bestvideo[height<=360][ext=mp4]/bestvideo[height<=360]",
        "outtmpl": video_path,
        "quiet": True,
        "no_warnings": True,
        "no_progress_bar": True,
        "ffmpeg_location": ffmpeg_dir,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(video_url, download=True)
    except Exception as e:
        raise RuntimeError(f"视频下载失败: {e}")

    # 查找实际下载的文件
    if not os.path.exists(video_path):
        for f in os.listdir(output_dir):
            if f.endswith((".mp4", ".webm")):
                video_path = os.path.join(output_dir, f)
                break

    if not os.path.exists(video_path):
        raise RuntimeError("视频文件下载后未找到")

    # === 3. 提取 I 帧（关键帧） ===
    frame_pattern = os.path.join(output_dir, "keyframe_%05d.png")
    try:
        cmd = [
            ffmpeg_path,
            "-skip_frame", "nokey",       # 只解码关键帧（I 帧）
            "-i", video_path,
            "-vsync", "vfr",              # 可变帧率，只输出有帧的位置
            "-frame_pts", "1",            # 文件名使用 PTS（便于算时间戳）
            "-qscale:v", "2",             # 高质量 PNG
            "-y",
            frame_pattern,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        os.remove(video_path)
        raise RuntimeError(f"关键帧提取失败: {e.stderr}")

    # 收集帧文件
    frame_paths = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("keyframe_") and f.endswith(".png")
    ])

    # === 兜底：场景检测（skip_frame 未产出帧时） ===
    if not frame_paths:
        try:
            scene_pattern = os.path.join(output_dir, "scene_%05d.png")
            cmd2 = [
                ffmpeg_path,
                "-i", video_path,
                "-vf", "select=gt(scene\\,0.4)",
                "-vsync", "vfr",
                "-qscale:v", "2",
                "-y",
                scene_pattern,
            ]
            subprocess.run(cmd2, capture_output=True, text=True, check=True)
            frame_paths = sorted([
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.startswith("scene_") and f.endswith(".png")
            ])
        except Exception:
            pass

    if not frame_paths:
        os.remove(video_path)
        raise RuntimeError("未能从视频中提取到任何关键帧")

    print(f"  ✓ 原始关键帧: {len(frame_paths)} 个", file=sys.stderr)

    # === 4. 感知哈希去重 ===
    unique_frames = _deduplicate_frames(frame_paths)
    dup_count = len(frame_paths) - len(unique_frames)
    if dup_count > 0:
        print(f"  ✓ 去重移除: {dup_count} 个相似帧", file=sys.stderr)

    # === 5. 均匀采样 ===
    frames = _select_frames_evenly(unique_frames, duration, max_frames)
    if len(frames) < len(unique_frames):
        print(f"  ✓ 采样保留: {len(frames)} 帧（上限 {max_frames}）", file=sys.stderr)

    # === 6. 清理视频文件 ===
    try:
        os.remove(video_path)
    except Exception:
        pass

    frames.sort(key=lambda f: f["timestamp"])
    return frames


def describe_frames_with_gemini(frames: list, api_key: str = None,
                                 transcript_summary: str = "",
                                 model_name: str = "gemini-1.5-pro") -> str:
    """
    使用 Gemini 多模态模型分析关键帧画面内容。

    Args:
        frames: extract_frames() 返回的帧列表
        api_key: Gemini API Key（可选，可从 config.yaml 或环境变量读取）
        transcript_summary: 字幕/转写文本摘要（供 Gemini 参考）
        model_name: Gemini 模型名称

    Returns:
        str: 视觉内容的结构化描述文本

    Raises:
        ValueError: API Key 未配置
        ImportError: google-generativeai 未安装
        RuntimeError: Gemini API 调用失败
    """
    # 解析 API Key
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Gemini API Key 未配置。请通过以下方式之一设置：\n"
            "  1. config.yaml → gemini.api_key\n"
            "  2. 环境变量: GEMINI_API_KEY=your-key\n"
            "  3. 命令行: --gemini-key your-key"
        )

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai 未安装。请运行: pip install google-generativeai"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # 构建多模态 Prompt
    text_prompt = (
        "You are analyzing keyframes extracted from a YouTube video.\n\n"
    )
    if transcript_summary:
        text_prompt += (
            "## Transcript Context\n"
            f"{transcript_summary}\n\n"
        )
    text_prompt += (
        "## Task\n"
        "Analyze the visual content shown in these keyframes. Describe:\n"
        "1. Type of visual content (slides, code, diagrams, charts, live footage, etc.)\n"
        "2. Visible text (slide titles, on-screen text, code snippets, captions)\n"
        "3. Notable visual elements that ADD information beyond the transcript\n"
        "4. Scene transitions or topic changes visible across frames\n\n"
        "Be specific and factual. If frame quality is poor, say so.\n"
        "Focus on information value — what can you SEE that the transcript might not capture?"
    )

    # 组装内容列表：文本 + 多张图片
    prompt_parts = [text_prompt]
    for frame in frames:
        ts = frame.get("timestamp", 0)
        # 添加时间戳标注
        prompt_parts.append(f"\n[Frame at {ts // 60:.0f}:{ts % 60:02.0f}]\n")
        try:
            img = Image.open(frame["path"])
            prompt_parts.append(img)
        except Exception:
            prompt_parts.append("(image unavailable)")

    # 调用 Gemini
    try:
        response = model.generate_content(
            prompt_parts,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
            }
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API 调用失败: {e}")


def cleanup_frames(frames: list):
    """
    删除临时帧文件及其空父目录。

    Args:
        frames: extract_frames() 返回的帧列表
    """
    if not frames:
        return

    dirs = set()
    for frame in frames:
        path = frame.get("path", "")
        if path and os.path.exists(path):
            try:
                os.remove(path)
                dirs.add(os.path.dirname(path))
            except Exception:
                pass

    for d in dirs:
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        except Exception:
            pass
