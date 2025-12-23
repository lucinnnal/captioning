import argparse
import json
import re
from pathlib import Path

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def extract_after_think(text: str) -> str:
    if text is None:
        return ""
    idx = text.find("</think>")
    if idx == -1:
        return text.strip()
    return text[idx + len("</think>"):].strip()

def normalize_text(text: str) -> str:
    # keep only post-</think> content, then flatten markdown-ish newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", " ", text)              # newlines -> spaces
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace
    return text

def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    # fallback: if no punctuation split happened, split by "  " or just return as one
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, type=Path)
    ap.add_argument("--out_jsonl", required=True, type=Path)
    ap.add_argument("--video_base_dir", default="", type=str,
                    help="e.g., /Users/changbeenkim (output video path = base_dir/video_name)")
    args = ap.parse_args()

    video_base = Path(args.video_base_dir) if args.video_base_dir else None

    with args.in_jsonl.open("r", encoding="utf-8") as rf, \
         args.out_jsonl.open("w", encoding="utf-8") as wf:
        for line_no, line in enumerate(rf, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            video_name = obj.get("video_name", "")
            if not video_name:
                raise ValueError(f"[line {line_no}] missing 'video_name'")

            caption = obj.get("caption", "")
            post = extract_after_think(caption)
            post = normalize_text(post)

            sents = split_sentences(post)
            sent_objs = [{"id": f"S{i+1}", "text": s} for i, s in enumerate(sents)]

            video_path = str((video_base / video_name) if video_base else video_name)

            out = {
                "data": {
                    "video": video_path,
                    "sentences": sent_objs
                }
            }
            wf.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

if __name__ == "__main__":
    main()