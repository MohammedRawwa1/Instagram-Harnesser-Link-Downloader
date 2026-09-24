# groq/ — standalone Groq transcription CLI

This folder is the same battle-tested CLI that powers the backend's
long-file path. Fully self-contained: point it at a file or a folder and it
produces `.txt` transcripts and — with `--srt` — lip-sync-accurate `.srt`
subtitles built from **word-level timestamps** and overlapping chunks.

## Setup

```bash
cd groq
pip install -r requirements.txt
cp .env.example .env      # then paste your GROQ_API_KEY
```

## Usage

```bash
python transcribe.py inputs/video.mp4            # transcript only
python transcribe.py inputs/ --srt               # folder batch, with subtitles
python transcribe.py inputs/reel.mp4 --srt --force
```

Outputs land in `transcripts/` (mirroring `inputs/` subfolders):

- `transcript - <name>.txt` — clean reading transcript
- `subtitle - <name>.srt`   — word-level, cue-engineered subtitles (with `--srt`)

## How the sync works

1. Audio is compressed to 16 kHz mono MP3.
2. Long files are split into **overlapping** chunks so no word is ever cut in
   half at a boundary (overlapping audio is deduped at stitch time).
3. Every chunk is transcribed with `timestamp_granularities=["word","segment"]`
   so each word carries exact start/end times.
4. Words are grouped into cues: max 42 chars, max 7 s on screen, breaks at
   pauses > 0.35 s, end-padding that never bleeds into the next cue.