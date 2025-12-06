# LLM Driven Podcast Generator

Generate a two-host educational podcast from lecture materials (PDF, PPTX, or TXT). The pipeline runs three phases: research & enrichment, scriptwriting, and audio production with OpenAI TTS. You can swap LLM providers (OpenAI or Groq), resume from intermediate artifacts and customize voices/timing in `config.py`.

## Features
- Multi-format ingest (PDF, PPTX, TXT) with slide/page context
- Gap analysis + optional web search (DDGS) to enrich the source material
- Persona-driven scriptwriter that outputs human-readable text and JSON for TTS
- Audio producer with OpenAI TTS, noise reduction, compression, and caching
- Resumable phases: start from existing enriched content or script JSON
- Tunable presets (draft/standard/premium) and detailed knobs in `config.py`

## Requirements
- Python 3.10+
- FFmpeg installed and on PATH (required by `pydub` for audio)
- Accounts/keys: OpenAI (LLM+TTS), optional Groq (LLM), optional ElevenLabs (not used by default)

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment
Create `.env` with your keys:
```bash
OPENAI_API_KEY=your-openai-key
GROQ_API_KEY=your-groq-key           # optional unless using --llm-provider groq
OUTPUT_DIR=./data/outputs
QUALITY_PRESET=premium               # draft | standard | premium
```

## Usage
From the project root after activating the venv:
```bash
# Full pipeline (OpenAI by default)
python main.py path/to/lecture.pdf --title "Quantum Basics" --notes "Focus on applications"

# Use Groq for the alternative LLM
python main.py lecture.pdf --llm-provider groq --groq-key %GROQ_API_KEY%

# Script only (skip audio)
python main.py lecture.pptx --script-only

# Resume from existing enriched content
python main.py --from-enriched data/outputs/enriched_content/enriched_20241206.txt --title "My Podcast"

# Generate audio from an existing script JSON
python main.py --from-script data/outputs/scripts/script_20241206.json

# Custom output directory
python main.py lecture.pdf --output-dir ./my_outputs
```

OpenAI key is always required for TTS; for script-only runs you can omit it if you pass `--script-only`.

## Outputs
- `data/outputs/enriched_content/` enriched notes (`enriched_*.txt`)
- `data/outputs/scripts/` script text + JSON for TTS (`script_*.txt/json`)
- `data/outputs/audio/` final mixed podcast MP3 (`podcast_*.mp3`)
- `data/outputs/audio_cache/` cached TTS clips (if enabled)
- `data/outputs/results_*.json` run summary (paths, title, provider)

## Configuration
`config.py` controls LLM/TTS models, voice selections, pacing, overlap/interruptions, caching, and quality presets. Adjust there instead of editing the agents directly.

## Notes
- `.gitignore` already excludes `.env`, `data/`, caches, and bytecode.
- Make sure FFmpeg is installed if audio generation fails.
- License: Apache-2.0 (`LICENSE`).
