"""
video-summarizer-skill 脚本包

模块清单：
    youtube.py         — YouTube 字幕提取 + 音频下载
    transcriber.py     — 语音识别转写
    chunker.py         — 文本分块（超长视频分段）
    frameextractor.py  — 视频关键帧提取 + Gemini 多模态分析
    skill.py           — 主入口
"""

from . import youtube
from . import transcriber
from . import chunker
from . import frameextractor
from . import skill

__all__ = ["youtube", "transcriber", "chunker", "frameextractor", "skill"]
