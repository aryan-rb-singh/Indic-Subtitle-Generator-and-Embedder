#  Indic Subtitle Generator — v1.1

> Automatically transcribe any video and generate subtitles in **all 22 scheduled Indian languages + English** — in one command.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

---

##  What it does

| Step | What happens |
|------|-------------|
| **1** | Extracts audio from the input video via `ffmpeg` |
| **2** | Transcribes speech using **OpenAI Whisper large-v3** (word-level timestamps) |
| **3** | Builds sync-corrected subtitle segments |
| **4** | Translates into **22 Indic languages + English** using **Meta NLLB-200** |
| **5** | Embeds all tracks as `.ass` subtitles inside a single `.mkv` output |

---

##  Supported Languages

Bengali · Telugu · Marathi · Tamil · Urdu · Gujarati · Kannada · Odia · Malayalam · Punjabi · Assamese · Maithili · Sanskrit · Nepali · Sindhi · Kashmiri · Konkani · Dogri · Santali · Meitei · Bhojpuri · **English**

---

##  Quick Start

### Option A — Google Colab (Recommended, Free GPU)

Download and Upload the .ipynb file on Google Collab, Then Run cells 1–6 in order. No local setup needed.

### Option B — Local Python

**1. Install dependencies**
```bash
pip install faster-whisper ctranslate2 transformers sentencepiece tqdm huggingface_hub
# macOS/Linux: brew install ffmpeg
# Ubuntu:      sudo apt install ffmpeg
```

**2. Run**
```bash
python indic_subtitle_generator.py --input /path/to/video.mp4 --output ./output
```

**3. Watch in VLC**
Open the output `.mkv` → Subtitles → Sub Track → pick any language.

---

##  Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.8 | Runtime |
| faster-whisper | latest | ASR transcription |
| ctranslate2 | latest | Fast NLLB inference |
| transformers | latest | NLLB tokenizer |
| sentencepiece | latest | Tokenization |
| ffmpeg | any | Audio/video processing |

> **GPU strongly recommended.** A T4 GPU (free on Colab) processes a 10-min video in ~12 min. CPU is ~10× slower.

---

##  Repository Structure

```
indic-subtitle-generator/
├── indic_subtitle_generator.py     # Standalone Python script (CLI)
├── indic_subtitle_v6_COLAB.ipynb   # Google Colab notebook
├── indic_subtitle_v6_LOCAL.ipynb   # Local Jupyter notebook
├── requirements.txt
├── LICENSE
└── README.md
```

---

##  CLI Reference

```
usage: indic_subtitle_generator.py [-h] --input INPUT [--output OUTPUT] [--model-cache MODEL_CACHE]

optional arguments:
  -h, --help            Show this help message and exit
  --input  / -i         Path to input video file (.mp4 .mkv .avi .mov .webm .m4v)
  --output / -o         Output directory (default: ./output)
  --model-cache         Directory to cache downloaded models
```

---

##  Output

```
output/
└── MyVideo_subtitles/
    ├── MyVideo_all_subtitles.mkv       ← final MKV with all 23 tracks embedded
    ├── MyVideo_hindi_original.ass      ← original language
    ├── MyVideo_bengali.ass
    ├── MyVideo_telugu.ass
    ├── ...                             ← one .ass file per language
    └── MyVideo_audio.wav               ← extracted audio (intermediate)
```

---

##  Typical Processing Times (T4 GPU)

| Video Length | Approx. Time |
|---|---|
| 10 minutes | ~12 minutes |
| 30 minutes | ~35 minutes |
| 60 minutes | ~70 minutes |

---

##  Models Used

- **[OpenAI Whisper large-v3](https://huggingface.co/openai/whisper-large-v3)** — ASR, auto-detects source language
- **[Meta NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)** — Neural machine translation for 200 languages

Both models are downloaded automatically on first run and cached locally.

---

##  License

MIT — see [LICENSE](LICENSE).

---

##  Contributing

Pull requests are welcome! Please open an issue first for major changes.

---

##  Acknowledgements

Built on top of [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2), and [Hugging Face Transformers](https://github.com/huggingface/transformers).
