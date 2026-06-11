"""
语音识别 / 文本转写模块

V0.2 实现：
    - 使用 faster-whisper 本地模型将音频转写为文本
    - 支持多种模型大小（tiny/base/small/medium/large）
    - 支持 CPU / CUDA 运行
    - 输出结构与 fetch_transcript() 兼容，便于上层统一消费

V0.4 预留：
    - Whisper API 云端转写（备选方案）

faster-whisper 说明：
    - 基于 CTranslate2，相比原版 Whisper 速度提升约 4 倍，内存占用更低
    - 首次运行时会自动下载模型（约 150MB~3GB 不等）
    - 模型缓存目录：~/.cache/huggingface/hub/
"""

import os
import sys
import time
from typing import Optional


def get_available_models() -> list:
    """返回 faster-whisper 支持的模型列表"""
    return ["tiny", "base", "small", "medium", "large-v3"]


def get_model_size_mb(model_size: str) -> int:
    """估算各模型大小（MB），用于提示用户"""
    sizes = {
        "tiny": 150,
        "base": 300,
        "small": 500,
        "medium": 1500,
        "large-v3": 3000,
    }
    return sizes.get(model_size, 500)


def transcribe(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "default",
    language: str = "zh",
    verbose: bool = True,
) -> dict:
    """
    使用 faster-whisper 将音频文件转写为文本。

    Args:
        audio_path: 音频文件路径（支持 mp3, wav, m4a 等格式）
        model_size: 模型大小。
                    tiny (150MB) / base (300MB) / small (500MB) /
                    medium (1.5GB) / large-v3 (3GB)
                    中文内容建议使用 small 或以上
        device: 运行设备。cpu / cuda
        compute_type: 计算精度。cpu 建议 "default" 或 "int8"；
                      cuda 建议 "float16" 或 "int8_float16"
        language: 语言代码（如 "zh", "en"），None 表示自动检测
        verbose: 是否打印进度信息

    Returns:
        dict: {
            "source": "asr",
            "language": str,
            "language_code": str,
            "language_probability": float,
            "model_size": str,
            "device": str,
            "duration": float,          # 音频时长（秒）
            "transcription_time": float, # 转写耗时（秒）
            "snippets": [
                {"start": float, "duration": float, "text": str},
                ...
            ],
            "full_text": str,
        }

    Raises:
        ImportError: faster-whisper 未安装
        FileNotFoundError: 音频文件不存在
        RuntimeError: 转写失败
    """
    # ====== 检查依赖 ======
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper 未安装。请运行: pip install faster-whisper\n"
            "或使用 conda: conda install -c conda-forge ctranslate2 faster-whisper"
        )

    # ====== 检查文件 ======
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    model_size_mb = get_model_size_mb(model_size)

    if verbose:
        print(f"🎤 语音识别配置:", file=sys.stderr)
        print(f"   模型: {model_size} (~{model_size_mb}MB)", file=sys.stderr)
        print(f"   设备: {device}", file=sys.stderr)
        print(f"   语言: {language or '自动检测'}", file=sys.stderr)
        print(f"   音频: {os.path.basename(audio_path)} ({file_size_mb:.1f}MB)", file=sys.stderr)
        print(f"   正在加载模型（首次使用会自动下载）...", file=sys.stderr)
        sys.stderr.flush()

    # ====== 加载模型 ======
    start_time = time.time()
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            # CPU 上 int8 能显著提升速度
            **({"cpu_threads": os.cpu_count() or 4} if device == "cpu" else {}),
        )
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}\n"
                           f"提示：确保网络可访问 HuggingFace，或检查模型名是否正确。")

    if verbose:
        load_time = time.time() - start_time
        print(f"   ✓ 模型加载完成 ({load_time:.1f}s)", file=sys.stderr)
        print(f"   正在转写音频...", file=sys.stderr)
        sys.stderr.flush()

    # ====== 执行转写 ======
    transcribe_start = time.time()
    try:
        # segments 是生成器，分段逐个返回结果
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,          # 启用 VAD 过滤静音段
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        # 收集结果
        detected_language = info.language
        detected_probability = info.language_probability

        snippets = []
        full_text_parts = []
        segment_count = 0

        for seg in segments:
            segment_count += 1
            duration = seg.end - seg.start
            snippets.append({
                "start": seg.start,
                "duration": duration,
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

            # 进度报告（每 50 段或最后一段）
            if verbose and segment_count % 50 == 0:
                print(f"   已转写 {segment_count} 段 "
                      f"(进度: {_fmt_ts(seg.end)} / {_fmt_ts(info.duration or 0)})",
                      file=sys.stderr)
                sys.stderr.flush()

    except Exception as e:
        raise RuntimeError(f"转写失败: {e}")

    transcribe_time = time.time() - transcribe_start

    if verbose:
        print(f"   ✓ 转写完成！共 {segment_count} 段", file=sys.stderr)
        print(f"   ✓ 检测语言: {detected_language} "
              f"(置信度: {detected_probability:.1%})", file=sys.stderr)
        print(f"   ⏱ 转写耗时: {transcribe_time:.1f}s", file=sys.stderr)
        sys.stderr.flush()

    return {
        "source": "asr",
        "language": detected_language,
        "language_code": detected_language,
        "language_probability": detected_probability,
        "model_size": model_size,
        "device": device,
        "duration": info.duration if hasattr(info, 'duration') and info.duration else 0,
        "transcription_time": transcribe_time,
        "snippets": snippets,
        "full_text": "\n".join(full_text_parts),
    }


def transcribe_with_api(audio_path: str, api_key: str = None,
                        model: str = "whisper-1") -> dict:
    """
    通过 Whisper API 转写音频（预留云端备选方案）。

    适用于：
        - 本地设备性能不足
        - 需要更高识别准确率
        - 极长视频转写

    Args:
        audio_path: 音频文件路径
        api_key: OpenAI API Key（也可从环境变量 OPENAI_API_KEY 读取）
        model: Whisper 模型名

    Returns:
        dict: 同 transcribe() 输出格式

    Raises:
        NotImplementedError: 功能预留，尚未实现
    """
    raise NotImplementedError(
        "Whisper API 转写功能预留。如需使用，请配置 OPENAI_API_KEY。"
    )


def _fmt_ts(seconds: float) -> str:
    """将秒数格式化为 MM:SS 或 H:MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
