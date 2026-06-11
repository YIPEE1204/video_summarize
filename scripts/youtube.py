"""
YouTube 视频处理模块

提供 YouTube 视频字幕提取 + 音频下载功能的封装。
V0.1 实现：字幕提取（有字幕场景）
V0.2 新增：音频流下载（无字幕场景，供 ASR 转写）
V0.4 预留：视频信息获取

API 用法说明（youtube_transcript_api v3）：
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=['zh', 'en'])
    # transcript 可迭代，每个 snippet 有 .start, .duration, .text 属性
    # transcript.language, transcript.language_code 为字幕元信息
    transcript_list = api.list(video_id)
    # 每个 t 有 .language_code, .language, .is_generated 属性
"""

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import re
import os
import tempfile
import sys


def extract_video_id(input_str: str) -> str:
    """
    从 URL 或纯 ID 中提取视频 ID。

    支持的格式：
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - 纯 11 位视频 ID
    """
    patterns = [
        r'youtube\.com/watch\?v=([\w-]+)',
        r'youtu\.be/([\w-]+)',
        r'^([\w-]{11})$',
    ]
    for p in patterns:
        match = re.search(p, input_str)
        if match:
            return match.group(1)
    raise ValueError("无法识别的视频 URL 或 ID")


def fetch_transcript(video_id: str, languages: list = None) -> dict:
    """
    获取视频字幕并返回结构化数据。

    Args:
        video_id: YouTube 视频 ID
        languages: 优先语言列表，按优先级排列。
                   默认 ['zh-Hans', 'zh', 'en']

    Returns:
        dict: {
            "video_id": str,
            "language": str,            # 如 "English"
            "language_code": str,       # 如 "en"
            "is_generated": bool,       # 是否为自动生成字幕
            "snippets": [               # 字幕片段列表
                {"start": float, "duration": float, "text": str},
                ...
            ],
            "full_text": str,           # 纯文本拼接（无时间戳）
        }

    Raises:
        TranscriptsDisabled: 视频没有字幕
        NoTranscriptFound: 找不到匹配语言的字幕
        VideoUnavailable: 视频不可用
    """
    if languages is None:
        languages = ['zh-Hans', 'zh', 'en']

    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=languages)

    snippets = []
    full_text_parts = []
    for snippet in transcript:
        snippets.append({
            "start": snippet.start,
            "duration": snippet.duration,
            "text": snippet.text,
        })
        full_text_parts.append(snippet.text)

    return {
        "video_id": video_id,
        "language": transcript.language,
        "language_code": transcript.language_code,
        "is_generated": getattr(transcript, 'is_generated', False),
        "snippets": snippets,
        "full_text": "\n".join(full_text_parts),
    }


def list_languages(video_id: str) -> list:
    """
    列出视频所有可用字幕语言。

    Returns:
        list[dict]: [
            {"language_code": str, "language": str, "is_generated": bool},
            ...
        ]
    """
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    result = []
    for t in transcript_list:
        result.append({
            "language_code": t.language_code,
            "language": t.language,
            "is_generated": t.is_generated,
        })
    return result


def save_transcript_to_file(video_id: str, languages: list = None,
                            output_path: str = None) -> str:
    """
    获取字幕并保存为 txt 文件。

    Args:
        video_id: YouTube 视频 ID
        languages: 优先语言列表
        output_path: 输出文件路径，默认 "{video_id}_transcript.txt"

    Returns:
        str: 保存的文件路径
    """
    data = fetch_transcript(video_id, languages)

    if output_path is None:
        output_path = f"{video_id}_transcript.txt"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"视频: https://www.youtube.com/watch?v={data['video_id']}\n")
        f.write(f"语言: {data['language']} ({data['language_code']})\n")
        f.write(f"字幕类型: {'自动生成' if data['is_generated'] else '手动上传'}\n")
        f.write(f"字幕条数: {len(data['snippets'])}\n")
        f.write("=" * 60 + "\n\n")

        for snippet in data['snippets']:
            start = snippet['start']
            end = start + snippet['duration']
            f.write(f"[{_fmt_time(start)} -> {_fmt_time(end)}] {snippet['text']}\n")

    print(f"✓ 已保存字幕到: {output_path}")
    return output_path


def format_transcript_text(data: dict, include_timestamps: bool = True) -> str:
    """
    将字幕数据格式化为可读文本。

    Args:
        data: fetch_transcript() 返回的字典
        include_timestamps: 是否包含时间戳

    Returns:
        str: 格式化后的文本
    """
    lines = [
        f"视频: https://www.youtube.com/watch?v={data['video_id']}",
        f"语言: {data['language']} ({data['language_code']})",
        f"字幕类型: {'自动生成' if data['is_generated'] else '手动上传'}",
        f"字幕条数: {len(data['snippets'])}",
        "=" * 60,
        "",
    ]

    if include_timestamps:
        for snippet in data['snippets']:
            start = snippet['start']
            end = start + snippet['duration']
            lines.append(f"[{_fmt_time(start)} -> {_fmt_time(end)}] {snippet['text']}")
    else:
        lines.append(data['full_text'])

    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    """将秒数格式化为 MM:SS 或 HH:MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def get_video_info(video_id: str) -> dict:
    """
    获取视频基本信息（标题、时长等），使用 yt-dlp。

    Args:
        video_id: YouTube 视频 ID

    Returns:
        dict: {
            "title": str,
            "duration": float,   # 秒
            "uploader": str,
        }
    """
    try:
        import yt_dlp
    except ImportError:
        return {"title": video_id, "duration": 0, "uploader": ""}

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", video_id),
                "duration": info.get("duration", 0),
                "uploader": info.get("uploader", ""),
            }
    except Exception:
        return {"title": video_id, "duration": 0, "uploader": ""}


def _find_ffmpeg() -> str:
    """
    自动查找系统中可用的 ffmpeg 路径。

    搜索顺序：
        1. conda 环境下的 Library/bin
        2. 当前 Python 解释器所在目录
        3. PATH 环境变量
        4. 常见安装路径

    Returns:
        str: ffmpeg 所在目录路径，未找到则返回空字符串
    """
    # 候选路径列表
    candidates = []

    # 1. conda 环境路径（Library/bin 在 conda env 目录下）
    python_dir = os.path.dirname(sys.executable)
    conda_lib_bin = os.path.join(python_dir, "Library", "bin")
    candidates.append(conda_lib_bin)
    candidates.append(python_dir)

    # 2. 常见安装路径
    common_paths = [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        os.path.expanduser("~/ffmpeg/bin"),
    ]
    candidates.extend(common_paths)

    # 3. PATH 环境变量中的路径
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    candidates.extend(path_dirs)

    # 去重并检查
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        ffmpeg_path = os.path.join(path, "ffmpeg.exe")
        if os.path.exists(ffmpeg_path):
            return path

    return ""


def download_audio(video_id: str, output_dir: str = None) -> dict:
    """
    使用 yt-dlp 下载视频的最佳音频流，保存为临时文件。

    Args:
        video_id: YouTube 视频 ID
        output_dir: 输出目录，默认使用系统临时目录

    Returns:
        dict: {
            "file_path": str,        # 音频文件路径（.mp3）
            "duration": float,       # 音频时长（秒）
            "title": str,            # 视频标题
        }

    Raises:
        ImportError: yt-dlp 未安装
        RuntimeError: 下载失败
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp 未安装，无法下载音频。请运行: pip install yt-dlp"
        )

    url = f"https://www.youtube.com/watch?v={video_id}"

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="ytaudio_")

    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    # 自动查找 ffmpeg
    ffmpeg_dir = _find_ffmpeg()
    if not ffmpeg_dir:
        raise RuntimeError(
            "未找到 ffmpeg，无法处理音频。\n"
            "请安装 ffmpeg：conda install -n summarize_env ffmpeg\n"
            "或下载 https://ffmpeg.org/download.html 并添加到 PATH"
        )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": ffmpeg_dir,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # 确定实际文件名
            ext = "wav"
            audio_path = os.path.join(output_dir, f"{video_id}.{ext}")

            # 如果 mp3 不存在，尝试查找下载的实际文件
            if not os.path.exists(audio_path):
                for f in os.listdir(output_dir):
                    if f.startswith(video_id):
                        audio_path = os.path.join(output_dir, f)
                        break

            if not os.path.exists(audio_path):
                raise RuntimeError("音频文件下载后未找到")

            return {
                "file_path": os.path.abspath(audio_path),
                "duration": info.get("duration", 0),
                "title": info.get("title", video_id),
            }

    except Exception as e:
        raise RuntimeError(f"音频下载失败: {e}")


def cleanup_audio(file_path: str):
    """
    删除临时音频文件及其所在目录（如果为空）。

    Args:
        file_path: download_audio() 返回的音频文件路径
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            parent = os.path.dirname(file_path)
            # 如果父目录为空则一并删除
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
    except Exception:
        pass


def main():
    """CLI 入口：交互式字幕下载"""
    print("=" * 50)
    print("  YouTube 字幕下载工具")
    print("=" * 50)

    raw = input("\n请输入视频 URL 或 ID: ").strip()
    try:
        video_id = extract_video_id(raw)
    except ValueError as e:
        print(f"错误: {e}")
        return

    print(f"\n视频 ID: {video_id}")
    print(f"链接: https://www.youtube.com/watch?v={video_id}\n")

    try:
        langs = list_languages(video_id)
        print(f"可用字幕语言 ({len(langs)} 种):")
        for t in langs:
            kind = "手动" if not t["is_generated"] else "自动"
            print(f"  {t['language_code']} ({t['language']}) - {kind}")
        print()

        data = fetch_transcript(video_id)
        print(f"✓ 获取到 {len(data['snippets'])} 条字幕")
        print(f"✓ 语言: {data['language']} ({data['language_code']})")

        save_transcript_to_file(video_id)
    except TranscriptsDisabled:
        print("该视频没有字幕 (Subtitles are disabled)")
        print("提示: 可使用 skill.py 自动触发语音识别转写")
    except NoTranscriptFound:
        print("找不到匹配语言的字幕")
    except VideoUnavailable:
        print("视频不可用（可能被删除或私密）")
    except Exception as e:
        print(f"获取失败: {e}")


if __name__ == "__main__":
    main()
