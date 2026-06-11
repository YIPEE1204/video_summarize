[README.md](https://github.com/user-attachments/files/28824612/README.md)
# video_summarize
A video summarization skill for Claude code

Summarize YouTube videos with Claude Code — automatically extracts subtitles or transcribes audio, with optional keyframe analysis for richer understanding.

## What is this?

This is a [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) skill that takes a YouTube URL and produces a structured content summary. It handles four progressively capable scenarios:

| Scenario | What happens |
|----------|-------------|
| **Captioned video** | Fetches subtitles directly via `youtube-transcript-api` — zero heavy dependencies |
| **No captions** | Downloads audio, runs local ASR via `faster-whisper` |
| **Long video** | Auto-chunks at ~40k tokens with `tiktoken`, outputs segment-by-segment for Claude to merge |
| **Visual enhancement** | Extracts keyframes via `yt-dlp` + `ffmpeg`, optionally feeds them to Gemini for multimodal analysis |

Output is always formatted text on stdout, ready for Claude to consume. Files are not written to disk unless `--keep` is specified.

## Key Capabilities

### Subtitle Extraction

- Pulls subtitles from YouTube's built-in API — lightweight, no video download needed
- Multi-language priority: `zh-Hans > zh > en` (configurable)
- Detects auto-generated vs. manual captions

### Speech Recognition

- Falls back to `faster-whisper` when no subtitles are available
- Local ASR, no cloud costs, no data leaving your machine
- Supports model sizes from `tiny` (150 MB) to `large-v3` (3 GB)
- Chinese language recognition with configurable language hint

### Long-Video Chunking

- Auto-detects when transcript exceeds token threshold (~40k tokens)
- Splits at subtitle boundaries, preserving timestamps
- Outputs "read per-segment → merge summary" instructions for Claude
- Customizable via `--max-tokens`

### Visual Enhancement

- Extracts I-frames (keyframes) from video — naturally captures scene/slide changes
- Perceptual hash deduplication removes near-identical frames
- Samples evenly across the video timeline
- **Optional** Gemini multimodal analysis: sends frames to Gemini 1.5 Pro for slide text, code, chart recognition
- No API key? Frames timeline is still included in the prompt for Claude's reference

## Getting Started

### Prerequisites: Anaconda / Miniconda

This project requires **Anaconda or Miniconda** to manage the Python environment and system dependencies (especially ffmpeg).

<details>
<summary><b>No Anaconda yet?</b></summary>

Download and install:
- **Miniconda** (lightweight, recommended): https://docs.anaconda.com/miniconda/
- **Anaconda** (full distribution): https://www.anaconda.com/download

After installation, open a new terminal and verify:

```bash
conda --version
```

</details>

### Setup (one-time)

Let Claude set up the environment automatically, or follow these steps:

```bash
# 1. Create and activate a dedicated conda environment
conda create -n summarize_env python=3.10 -y
conda activate summarize_env

# 2. Install ffmpeg (needed for ASR and keyframe extraction)
conda install ffmpeg -y

# 3. Install Python dependencies
pip install -r requirements.txt
```

> **Network note**: YouTube and HuggingFace (for ASR model downloads) may require a proxy in some regions. Set `HTTP_PROXY` / `HTTPS_PROXY` environment variables if needed.

### Using with Claude Code

Once installed as a skill (`~/.claude/skills/video-summarizer`):

```
/summarize https://www.youtube.com/watch?v=VIDEO_ID
```

Claude Code will:

1. Read `SKILL.md` for the workflow definition
2. Execute `scripts/skill.py` with your URL
3. Receive structured transcript + summary instructions on stdout
4. Generate a comprehensive summary covering all video content

Make sure the `summarize_env` environment is activated when running.

### Manual Usage

All scripts live in `scripts/` and can be run directly:

```bash
# Navigate to project root
cd /path/to/video-summarizer

# Basic — auto-detect captions or fall back to ASR
python scripts/skill.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Force ASR (skip subtitle check)
python scripts/skill.py "https://youtu.be/VIDEO_ID" --asr

# Save transcript to disk
python scripts/skill.py "https://www.youtube.com/watch?v=VIDEO_ID" --keep

# Visual enhancement — extract keyframes
python scripts/skill.py "https://www.youtube.com/watch?v=VIDEO_ID" --visual

# Visual + Gemini multimodal analysis
python scripts/skill.py "https://www.youtube.com/watch?v=VIDEO_ID" --visual --gemini-key sk-xxx

# Custom chunk threshold for long videos
python scripts/skill.py "https://youtu.be/VIDEO_ID" --max-tokens 30000

# JSON output for programmatic use
python scripts/skill.py "https://youtu.be/VIDEO_ID" --json
```

### All Parameters

| Argument | Purpose |
|----------|---------|
| `url` | YouTube URL or video ID **(required)** |
| `-o, --output` | Save transcript to specified path (implies `--keep`) |
| `--keep` | Save transcript to `output/` directory |
| `--lang zh en` | Subtitle language priority |
| `--asr` | Force speech recognition (skip captions) |
| `--no-asr` | Disable ASR fallback (fail if no captions) |
| `--max-tokens N` | Chunk threshold (default: 40000) |
| `--visual` | Enable keyframe extraction |
| `--gemini-key KEY` | Gemini API key for multimodal analysis |
| `--json` | Output structured JSON instead of formatted text |

## Output Format

stdout delivers a formatted prompt that Claude reads to produce the summary:

```
============================================================
  YouTube Video Summary
============================================================

Video: https://www.youtube.com/watch?v=abc123
Language: English (en)
Source: Subtitles
Segments: 156

------------------------------------------------------------
  Full Transcript (with timestamps)
------------------------------------------------------------

[0:05 -> 0:10] Hello everyone, welcome to this video...
...

------------------------------------------------------------
  Summary Instructions
------------------------------------------------------------

Based on the transcript above, generate a structured summary...
```

- **stdout** — formatted transcript + instructions for Claude
- **stderr** — progress info visible to the user only
- **`--json`** — raw structured data for programmatic consumption

## How It Works

The skill follows an automatic degradation pipeline:

```
                    ┌──────────────┐
                    │  YouTube URL │
                    └──────┬───────┘
                           ▼
                 Fetch subtitles ────→ Found → youtube-transcript-api
                           │
                           ▼ (Not found)
                 Speech recognition ──→ faster-whisper (local)
                           │
                           ▼
                  Check token count ──→ Over threshold → tiktoken chunk
                           │
                           ▼ (--visual)
                  Keyframe extraction ──→ yt-dlp + ffmpeg I-frame
                           │
                           ▼ (Gemini key set)
                  Multimodal analysis ──→ Gemini 1.5 Pro
                           │
                           ▼
                  stdout: formatted summary prompt
```

Each step is optional and only triggers when needed. A captioned video never downloads audio or video.

## Configuration

Edit `config.yaml` in the project root:

```yaml
whisper:
  model_size: "base"        # tiny / base / small / medium / large-v3
  device: "cpu"             # cpu / cuda

chunker:
  max_tokens_per_chunk: 40000

gemini:
  api_key: ""               # Set here or via GEMINI_API_KEY env var
  model: "gemini-1.5-pro"
  max_frames: 10

output:
  directory: "output"       # Where --keep saves files
```

Proxy can be configured via environment variables:

```bash
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=https://127.0.0.1:7890
```

## Repository Structure

```
├── requirements.txt         # Python dependencies
├── config.yaml              # Global configuration
├── SKILL.md                 # Skill registration & workflow
├── README.md                # This file
├── output/                  # Saved transcripts (--keep)
└── scripts/                 # All Python source
    ├── skill.py             # Entry point — run this
    ├── __init__.py          # Package init
    ├── youtube.py           # Subtitle extraction + audio download
    ├── transcriber.py       # Speech recognition (faster-whisper)
    ├── chunker.py           # Text chunking (tiktoken)
    └── frameextractor.py    # Keyframe extraction + Gemini
```

## Notes

- **Legal compliance**: For personal study and research only. Respect platform terms of service.
- **Progressive dependencies**: Captioned videos need only `youtube-transcript-api`. ASR adds `faster-whisper`, `yt-dlp`, `ffmpeg`. Visual adds `Pillow`, optionally `google-generativeai`.
- **No disk writes by default**: Use `--keep` to persist. All generated content goes to stdout.
- **Gemini key is optional**: Without it, visual mode shows frame timestamps only; with it, Gemini describes visual content.
- **ffmpeg auto-detection**: The script finds ffmpeg in conda `Library/bin/`, Python directory, and PATH automatically.

## Attribution

This skill was developed as a phased project from V0.1 (subtitle extraction) through V0.4 (visual enhancement). Core technologies: `youtube-transcript-api`, `faster-whisper`, `yt-dlp`, `ffmpeg`, `tiktoken`, `Pillow`, and optionally `google-generativeai` for Gemini integration.
