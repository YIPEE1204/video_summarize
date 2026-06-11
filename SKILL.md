---
name: summarize
description: "Summarize YouTube video content — transcript + ASR + visual analysis"
---

# YouTube Video Summarizer Skill (`/summarize`)

Takes a YouTube video URL, extracts subtitles or transcribes audio, and outputs structured content for Claude to summarize. Supports four scenarios: captioned videos, no-caption ASR fallback, long-video chunking, and optional keyframe visual enhancement.

## Quick Start

### Prerequisites: Anaconda / Miniconda

Requires **Anaconda or Miniconda** to manage the Python environment and system dependencies (especially ffmpeg).

<details>
<summary><b>No Anaconda yet?</b></summary>

Download:
- **Miniconda** (lightweight, recommended): https://docs.anaconda.com/miniconda/
- **Anaconda** (full distribution): https://www.anaconda.com/download

After installation, verify in a new terminal:

```bash
conda --version
```

</details>

### 1. Create environment & install dependencies

```bash
# Create and activate a dedicated conda environment
conda create -n summarize_env python=3.10 -y
conda activate summarize_env

# Install ffmpeg (required for ASR and keyframe extraction)
conda install ffmpeg -y

# Install Python dependencies
pip install -r requirements.txt
```

> The first ASR run downloads a Whisper model (~300MB~3GB). Ensure your network can reach HuggingFace.
> Activate `summarize_env` before each run: `conda activate summarize_env`.

### 2. Usage

```
/summarize https://www.youtube.com/watch?v=VIDEO_ID
```

---

## Output Format

stdout delivers a formatted prompt that Claude reads directly to produce the summary:

```
============================================================
  YouTube Video Summary
============================================================

Video: https://www.youtube.com/watch?v=abc123
Language: English (en)
Source: Subtitles
Segments: 156
Duration: ~15:23

------------------------------------------------------------
  Full Transcript (with timestamps)
------------------------------------------------------------

[0:05 -> 0:10] Hello everyone...
...

------------------------------------------------------------
  Summary Instructions
------------------------------------------------------------

Based on the transcript above, generate a structured summary...
```

- **stdout** — formatted transcript + summary instructions for Claude
- **stderr** — progress info (video parsing, subtitle status, ASR progress), visible to the user only
- **`--json`** — raw structured data for programmatic consumption

### Chunked Output

When the transcript exceeds ~40k tokens, it splits automatically at subtitle boundaries with per-segment timeline markers. The summary instruction uses a "read per segment → merge" pattern.

### Visual Enhancement

Use `--visual` to extract keyframes. With a Gemini API key, frame content analysis is appended to the summary; without it, only frame paths are listed.

---

## Parameters

| Argument | Purpose | Use case |
|----------|---------|----------|
| `url` | YouTube URL or video ID | Required |
| `-o, --output` | Save transcript to specified path | Persist output |
| `--keep` | Save transcript to disk (default: stdout only) | Persist output |
| `--lang zh en` | Subtitle language priority | Multi-language videos |
| `--asr` | Force speech recognition (skip subtitles) | Poor subtitle quality |
| `--no-asr` | Disable ASR fallback | Subtitles only |
| `--max-tokens 30000` | Custom chunk threshold | Very long videos |
| `--visual` | Enable keyframe extraction | Visual enhancement |
| `--gemini-key KEY` | Gemini API key for multimodal analysis | Multimodal analysis |
| `--json` | JSON formatted output | Debugging / programmatic use |
| `-h, --help` | Show help | — |

---

## Usage Examples

### Basic

```
# Captioned video → extract subtitles
/summarize https://www.youtube.com/watch?v=abc123

# No captions → automatic ASR fallback (faster-whisper)
/summarize https://www.youtube.com/watch?v=abc123

# Force ASR
/summarize https://www.youtube.com/watch?v=abc123 --asr
```

### Advanced

```
# Specify subtitle language preference
/summarize https://youtu.be/abc123 --lang zh en

# Custom chunk threshold (smaller = less context used per segment)
/summarize https://youtu.be/abc123 --max-tokens 30000

# Save transcript to disk
/summarize https://youtu.be/abc123 --keep

# Visual enhancement (extract keyframes)
/summarize https://youtu.be/abc123 --visual

# Visual + multimodal analysis
/summarize https://youtu.be/abc123 --visual --gemini-key YOUR_KEY
```

---

## Pipeline

The skill follows an automatic degradation pipeline:

1. **Captioned** → `youtube-transcript-api` (zero heavy dependencies)
2. **No captions** → `faster-whisper` local ASR (requires yt-dlp + ffmpeg)
3. **Long video** → `tiktoken` auto-chunking (>40k tokens)
4. **Visual** (optional) → `yt-dlp + ffmpeg` keyframe extraction + Gemini multimodal analysis

Each step is optional and only triggers when needed. A captioned video never downloads audio or video.

---

## Configuration

Edit `config.yaml` in the project root:

```yaml
whisper:
  model_size: "base"        # tiny/base/small/medium/large-v3
  device: "cpu"             # cpu / cuda

chunker:
  max_tokens_per_chunk: 40000

gemini:
  api_key: ""               # Leave empty to skip multimodal analysis
  model: "gemini-1.5-pro"
  max_frames: 10

output:
  directory: "output"       # Where --keep saves files
```

---

## Network & Proxy

- **YouTube access**: Required for subtitle fetching, audio/video download
- **HuggingFace access**: Required on first ASR run (Whisper model download)
- **Configuration**: Set via environment variables

```bash
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
```

---

## Notes

- ⚠️ **Legal compliance**: For personal study and research only. Respect platform terms of service.
- 📦 **Progressive dependencies**: Captioned videos need only `youtube-transcript-api`. ASR adds `yt-dlp`, `faster-whisper`, `ffmpeg`. Visual adds `Pillow`, optionally `google-generativeai`.
- 🖼️ **Visual mode is opt-in**: Use `--visual` explicitly. Requires Pillow.
- 🔑 **Gemini key is optional**: Without it, only frame paths are shown.
- 💾 **No disk writes by default**: Text goes to stdout only. Use `--keep` or `-o` to persist.
- 🔧 **ffmpeg auto-detection**: Finds ffmpeg in conda `Library/bin/`, Python directory, and PATH automatically.
- 🐍 **Python**: 3.10+ recommended.

---

## Project Structure

```
├── requirements.txt         # Python dependencies
├── config.yaml              # Global configuration
├── SKILL.md                 # Skill registration file
├── README.md                # User manual
├── output/                  # Output directory (--keep)
└── scripts/
    ├── skill.py             # Entry point
    ├── youtube.py           # Subtitle extraction + audio download
    ├── transcriber.py       # Speech recognition (faster-whisper)
    ├── chunker.py           # Text chunking (tiktoken)
    └── frameextractor.py    # Keyframe extraction + Gemini
```
