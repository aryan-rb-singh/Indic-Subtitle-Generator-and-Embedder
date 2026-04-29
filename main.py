#!/usr/bin/env python3
"""
Indic Subtitle Generator — v1.1
==============================
Automatically transcribes a video using OpenAI Whisper (large-v3),
then translates the subtitles into all 22 scheduled Indian languages
plus English using Meta NLLB-200, and embeds every track as ASS
subtitles inside a single .mkv output file.

Supported input formats: .mp4  .mkv  .avi  .mov  .webm  .m4v

Usage
-----
    # Install dependencies first:
    pip install faster-whisper ctranslate2 transformers sentencepiece tqdm huggingface_hub

    # Run:
    python indic_subtitle_generator.py --input /path/to/video.mp4 --output /path/to/output/

Requirements
------------
- Python 3.8+
- ffmpeg  (must be on PATH)
- CUDA GPU strongly recommended (CPU mode ~10x slower)

Author : (your name / handle)
License: MIT
"""

import os
import re
import json
import shutil
import subprocess
import time
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Optional tqdm (falls back to plain print) ────────────────────────────────
try:
    from tqdm import tqdm as _tqdm
    def tqdm(iterable, **kw):
        return _tqdm(iterable, **kw)
except ImportError:
    def tqdm(iterable, desc="", **kw):
        print(f"  {desc}...")
        return iterable

# ── Language tables ──────────────────────────────────────────────────────────
INDIC_LANGUAGES = {
    "Hindi":    "hin_Deva",  "Bengali":  "ben_Beng",  "Telugu":  "tel_Telu",
    "Marathi":  "mar_Deva",  "Tamil":    "tam_Taml",  "Urdu":    "urd_Arab",
    "Gujarati": "guj_Gujr",  "Kannada":  "kan_Knda",  "Odia":    "ory_Orya",
    "Malayalam":"mal_Mlym",  "Punjabi":  "pan_Guru",  "Assamese":"asm_Beng",
    "Maithili": "mai_Deva",  "Sanskrit": "san_Deva",  "Nepali":  "npi_Deva",
    "Sindhi":   "snd_Arab",  "Kashmiri": "kas_Arab",  "Konkani": "kok_Deva",
    "Dogri":    "doi_Deva",  "Santali":  "sat_Olck",  "Meitei":  "mni_Mtei",
    "Bhojpuri": "bho_Deva",  "English":  "eng_Latn",
}

WHISPER_TO_NLLB = {
    "hi": "hin_Deva", "en": "eng_Latn", "bn": "ben_Beng", "te": "tel_Telu",
    "mr": "mar_Deva", "ta": "tam_Taml", "ur": "urd_Arab", "gu": "guj_Gujr",
    "kn": "kan_Knda", "or": "ory_Orya", "ml": "mal_Mlym", "pa": "pan_Guru",
    "as": "asm_Beng", "ne": "npi_Deva", "sa": "san_Deva", "sd": "snd_Arab",
    "ks": "kas_Arab", "si": "sin_Sinh",
}

NLLB_TO_NAME = {v: k for k, v in INDIC_LANGUAGES.items()}

LANG_ISO = {
    "Hindi": "hin", "Bengali": "ben", "Telugu": "tel", "Marathi": "mar",
    "Tamil": "tam", "Urdu": "urd", "Gujarati": "guj", "Kannada": "kan",
    "Odia": "ori", "Malayalam": "mal", "Punjabi": "pan", "Assamese": "asm",
    "Maithili": "mai", "Sanskrit": "san", "Nepali": "nep", "Sindhi": "snd",
    "Kashmiri": "kas", "Konkani": "kok", "Dogri": "doi", "Santali": "sat",
    "Meitei": "mni", "Bhojpuri": "bho", "English": "eng",
}

# ── Tuning constants ─────────────────────────────────────────────────────────
SYNC_LEAD_MS       = 350
MAX_SUB_SECS       = 4.0
MIN_GAP_MS         = 50
MAX_CHARS_PER_LINE = 60

# ── ASS subtitle helpers ─────────────────────────────────────────────────────
def _ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    cs  = int(round(sec * 100)) % 100
    s   = int(sec) % 60
    m   = (int(sec) // 60) % 60
    h   = int(sec) // 3600
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes
WrapStyle: 0
Collisions: Normal

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_ass(segments: list, path: str) -> None:
    """Write a list of {start, end, text} dicts to an ASS subtitle file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(_ASS_HEADER)
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            text = re.sub(r"\s*\n\s*", " ", text).strip()
            if len(text) > MAX_CHARS_PER_LINE:
                text = text[:MAX_CHARS_PER_LINE].rsplit(" ", 1)[0] + "…"
            f.write(
                f"Dialogue: 0,{_ass_time(seg['start'])},{_ass_time(seg['end'])},"
                f"Default,,0,0,0,,{text}\n"
            )


def read_ass(path: str) -> list:
    """Parse an ASS file and return a list of {start, end, text} dicts."""
    segs, in_ev, fmt = [], False, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "[Events]":
                in_ev = True
                continue
            if in_ev:
                if line.startswith("Format:"):
                    fmt = [x.strip() for x in line[7:].split(",")]
                elif line.startswith("Dialogue:"):
                    parts = line[9:].split(",", len(fmt) - 1)
                    if len(parts) < len(fmt):
                        continue
                    d = dict(zip(fmt, parts))
                    try:
                        def _p(t):
                            t = t.strip()
                            h, m, sc = t.split(":")
                            s, cs = sc.split(".")
                            return int(h)*3600 + int(m)*60 + int(s) + int(cs)/100
                        segs.append({
                            "start": _p(d["Start"]),
                            "end":   _p(d["End"]),
                            "text":  d.get("Text", "").strip(),
                        })
                    except Exception:
                        pass
    return segs


def build_word_segments(raw_segments: list, lead_ms: int = SYNC_LEAD_MS) -> list:
    """Convert raw Whisper segments into display-ready subtitle lines."""
    lead_s    = lead_ms / 1000.0
    gap_s     = MIN_GAP_MS / 1000.0
    pause_thr = 0.30
    words = []
    for seg in raw_segments:
        if seg.get("words"):
            for w in seg["words"]:
                if w.word.strip():
                    words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
        else:
            words.append({"word": seg["text"].strip(), "start": seg["start"], "end": seg["end"]})
    if not words:
        return []

    groups, cur = [], [words[0]]
    for w in words[1:]:
        prev  = cur[-1]
        pause = w["start"] - prev["end"]
        dur   = w["end"] - cur[0]["start"]
        chars = sum(len(x["word"]) for x in cur) + len(cur) + len(w["word"]) + 1
        if dur > MAX_SUB_SECS or chars > MAX_CHARS_PER_LINE or pause > pause_thr:
            groups.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        groups.append(cur)

    segs = []
    for grp in groups:
        segs.append({
            "start": round(max(0.0, grp[0]["start"] - lead_s), 3),
            "end":   round(grp[-1]["end"], 3),
            "text":  " ".join(x["word"] for x in grp).strip(),
        })
    for i in range(len(segs) - 1):
        mx = segs[i + 1]["start"] - gap_s
        if segs[i]["end"] > mx:
            segs[i]["end"] = round(max(segs[i]["start"] + 0.1, mx), 3)
    return segs


# ── Translation helper ───────────────────────────────────────────────────────
def translate_batch(
    texts: list,
    src_lang: str,
    tgt_lang: str,
    tokenizer,
    translator,
    batch_size: int = 32,
    beam_size: int  = 2,
) -> list:
    """Translate a list of strings from src_lang to tgt_lang via NLLB."""
    if not texts:
        return []
    tokenizer.src_lang = src_lang
    results = []
    for i in range(0, len(texts), batch_size):
        batch     = texts[i : i + batch_size]
        valid_idx = [j for j, t in enumerate(batch) if t.strip()]
        valid_txt = [batch[j] for j in valid_idx]
        out       = list(batch)
        if not valid_txt:
            results.extend(out)
            continue
        try:
            enc = tokenizer(
                valid_txt, return_tensors=None,
                padding=False, truncation=True, max_length=256,
            )
            tb = [tokenizer.convert_ids_to_tokens(ids) for ids in enc["input_ids"]]
            tr = translator.translate_batch(
                tb, target_prefix=[[tgt_lang]] * len(tb),
                beam_size=beam_size, max_decoding_length=256,
                repetition_penalty=1.2, no_repeat_ngram_size=4,
            )
            for j, res in zip(valid_idx, tr):
                toks = res.hypotheses[0]
                if toks and toks[0] == tgt_lang:
                    toks = toks[1:]
                dec = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(toks),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                out[j] = dec if dec else batch[j]
        except Exception as e:
            print(f"  ⚠️  batch error: {e}")
        results.extend(out)
    return results


# ── ffprobe helper ───────────────────────────────────────────────────────────
def get_duration(video_path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", video_path],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        return float(json.loads(r.stdout).get("format", {}).get("duration", 0))
    return 0.0


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(video_path: str, output_dir: str, model_cache_dir: str = None):
    """
    Full pipeline:
      1. Extract audio
      2. Transcribe with Whisper large-v3
      3. Build word-level subtitle segments
      4. Translate into 22+ languages via NLLB-200
      5. Mux all ASS tracks into a single MKV
    """
    import torch

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    compute_ct = "float16" if device == "cuda" else "int8"

    video_path = str(Path(video_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    if model_cache_dir is None:
        model_cache_dir = os.path.join(output_dir, "_models")
    nllb_cache = os.path.join(model_cache_dir, "nllb_ct2_int8")
    os.makedirs(nllb_cache, exist_ok=True)

    video_name = Path(video_path).stem
    work_dir   = os.path.join(output_dir, f"{video_name}_subtitles")
    os.makedirs(work_dir, exist_ok=True)

    duration = get_duration(video_path)
    print("=" * 60)
    print(f"  🎬  {video_name}")
    print("=" * 60)
    print(f"   Duration : {int(duration)//60}m {int(duration)%60:02d}s")
    print(f"   Device   : {device}")
    print(f"   Output   : {work_dir}")

    # ── 1. Extract audio ─────────────────────────────────────────────────────
    print("\n── Step 1/5: Extracting audio ───────────────────────────")
    audio_path = os.path.join(work_dir, f"{video_name}_audio.wav")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Audio extraction failed:\n{r.stderr[-500:]}")
    print(f" Audio: {os.path.getsize(audio_path)/1024**2:.2f} MB")

    # ── 2. Load models ───────────────────────────────────────────────────────
    print("\n── Loading Whisper large-v3 ─────────────────────────────")
    from faster_whisper import WhisperModel
    import ctranslate2
    from ctranslate2.converters import TransformersConverter
    from transformers import NllbTokenizerFast

    t0 = time.time()
    whisper_model = WhisperModel("large-v3", device=device, compute_type=compute_ct)
    print(f" Whisper loaded in {time.time()-t0:.1f}s")

    print("\n── Loading NLLB-200-distilled-600M ──────────────────────")
    nllb_hf = "facebook/nllb-200-distilled-600M"
    if not os.path.exists(os.path.join(nllb_cache, "model.bin")):
        print("  First run: downloading + converting NLLB (~3-5 min)...")
        t0   = time.time()
        conv = TransformersConverter(nllb_hf, low_cpu_mem_usage=True)
        conv.convert(output_dir=nllb_cache, quantization="int8", force=True)
        print(f"   Converted in {time.time()-t0:.0f}s — cached to {nllb_cache}")
    else:
        print("   NLLB cache found — loading directly")

    ct2  = "int8_float16" if device == "cuda" else "int8"
    t0   = time.time()
    nllb_translator = ctranslate2.Translator(
        nllb_cache, device=device,
        compute_type=ct2, inter_threads=1, intra_threads=4,
    )
    nllb_tokenizer = NllbTokenizerFast.from_pretrained(nllb_hf)
    print(f" NLLB loaded in {time.time()-t0:.1f}s")

    # ── 3. Transcribe ────────────────────────────────────────────────────────
    print("\n── Step 2/5: Transcribing (word-level timestamps) ───────")
    t1 = time.time()
    segs_gen, meta = whisper_model.transcribe(
        audio_path, beam_size=5, language=None,
        word_timestamps=True,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 800,
            "speech_pad_ms": 400,
            "threshold": 0.35,
        },
    )
    detected_lang = meta.language
    src_nllb      = WHISPER_TO_NLLB.get(detected_lang, "hin_Deva")
    src_name      = NLLB_TO_NAME.get(src_nllb, f"Unknown ({detected_lang})")
    print(f"   Detected: {src_name} (\"{detected_lang}\")  |  {src_nllb}")

    raw = []
    for seg in segs_gen:
        raw.append({
            "start": seg.start, "end": seg.end,
            "text": seg.text, "words": seg.words,
        })
    print(
        f" Transcription done in {time.time()-t1:.1f}s — "
        f"{len(raw)} segments"
    )

    # ── 4. Build sync-corrected segments ─────────────────────────────────────
    print("\n── Step 3/5: Building sync-corrected subtitle segments ───")
    synced = build_word_segments(raw, lead_ms=SYNC_LEAD_MS)
    print(f" {len(synced)} subtitle lines")

    orig_slug    = src_name.split()[0].lower()
    original_ass = os.path.join(work_dir, f"{video_name}_{orig_slug}_original.ass")
    write_ass(synced, original_ass)
    print(f"   Saved original ASS: {orig_slug}_original.ass ({src_name})")

    # ── 5. Translate ─────────────────────────────────────────────────────────
    print(f"\n── Step 4/5: Translating to all {len(INDIC_LANGUAGES)-1} languages ────────")
    target_langs  = {n: c for n, c in INDIC_LANGUAGES.items() if c != src_nllb}
    orig_segs     = read_ass(original_ass)
    source_texts  = [s["text"] for s in orig_segs]
    generated_ass = []
    failed_langs  = []
    t_tr = time.time()

    for i, (lang_name, tgt_code) in enumerate(
        tqdm(list(target_langs.items()), desc="  Translating", ncols=70), 1
    ):
        print(f"  [{i:2d}/{len(target_langs)}] {lang_name:15s} ({tgt_code}) ", end="", flush=True)
        t0 = time.time()
        try:
            translated = translate_batch(
                source_texts, src_nllb, tgt_code, nllb_tokenizer, nllb_translator
            )
            trans_segs = [
                {"start": s["start"], "end": s["end"],
                 "text": t if t.strip() else s["text"]}
                for s, t in zip(orig_segs, translated)
            ]
            safe  = lang_name.lower().replace(" ", "_")
            apath = os.path.join(work_dir, f"{video_name}_{safe}.ass")
            write_ass(trans_segs, apath)
            generated_ass.append((lang_name, apath))
            print(f" {time.time()-t0:.1f}s")
        except Exception as e:
            print(f" {e}")
            failed_langs.append((lang_name, str(e)))

    print(f"\n Translation done in {time.time()-t_tr:.1f}s")
    if failed_langs:
        print(f"  Failed: {[l for l, _ in failed_langs]}")

    # ── 6. Mux into MKV ──────────────────────────────────────────────────────
    print(f"\n── Step 5/5: Embedding {1+len(generated_ass)} ASS tracks into MKV ────")
    all_tracks = []
    if os.path.exists(original_ass):
        all_tracks.append((
            f"Original-{src_name.split()[0]}",
            LANG_ISO.get(src_name.split()[0], "und"),
            original_ass,
        ))
    for lang_name, apath in generated_ass:
        all_tracks.append((lang_name, LANG_ISO.get(lang_name, "und"), apath))

    output_mkv = os.path.join(work_dir, f"{video_name}_all_subtitles.mkv")
    cmd = ["ffmpeg", "-y", "-i", video_path]
    for _, _, ass in all_tracks:
        cmd += ["-i", ass]
    cmd += ["-map", "0:v", "-map", "0:a?"]
    for k in range(len(all_tracks)):
        cmd += ["-map", f"{k+1}:0"]
    cmd += ["-c:v", "copy", "-c:a", "copy", "-c:s", "ass"]
    for k, (name, iso, _) in enumerate(all_tracks):
        cmd += [f"-metadata:s:s:{k}", f"title={name}"]
        cmd += [f"-metadata:s:s:{k}", f"language={iso}"]
    cmd += ["-metadata", f"title={video_name} — Multi-language Subtitles"]
    cmd += ["-disposition:s:0", "default", output_mkv]

    t0 = time.time()
    r  = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{r.stderr[-800:]}")

    out_mb = os.path.getsize(output_mkv) / 1024**2
    print(f" MKV muxed in {time.time()-t0:.1f}s")
    print()
    print("=" * 60)
    print("    PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Output MKV : {output_mkv}  ({out_mb:.1f} MB)")
    print(f"  Tracks     : {len(all_tracks)} subtitle languages (ASS)")
    print()
    print("   VLC: Subtitles menu → Sub Track → pick language")
    print("   MPV: '#' key to cycle tracks")
    return output_mkv


# ── CLI entry-point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-language Indic subtitles for a video file."
    )
    parser.add_argument("--input",  "-i", required=True,  help="Path to input video file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--model-cache", default=None,
                        help="Directory to cache downloaded models (default: <output>/_models)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f" Input file not found: {args.input}")
        raise SystemExit(1)

    run_pipeline(args.input, args.output, args.model_cache)


if __name__ == "__main__":
    main()
