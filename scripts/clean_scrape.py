#!/usr/bin/env python3
import re, json, unicodedata, argparse, hashlib, sys
from pathlib import Path

MD_IMAGE = re.compile(r'!\[[^\]]*\]\([^)]+\)')
MD_LINK  = re.compile(r'\[([^\]]+)\]\((?:[^)]+)\)')
MD_CODEBLOCK = re.compile(r'```.*?```', re.S)
MD_INLINECODE = re.compile(r'`[^`]+`')
MD_TABLE_LINE = re.compile(r'^\s*\|.*\|\s*$', re.M)
MD_FRONTMATTER = re.compile(r'^---\s*\n.*?\n---\s*\n', re.S)
MD_BLOCKQUOTE = re.compile(r'^\s*>\s?', re.M)
MD_BOLD_ITALIC = re.compile(r'[*_]{1,3}(.+?)[*_]{1,3}')
MD_HEADINGS = re.compile(r'^\s{0,3}#{1,6}\s*', re.M)
URL_RE   = re.compile(r'https?://\S+|www\.\S+', re.I)
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
EMOJI_RE = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
BRACKETS = re.compile(r'\[\s*\d+\s*\]')
ZERO_WIDTH = dict.fromkeys(map(ord, ["\u200b","\u200c","\u200d"]), None)

BANNED_LINE = re.compile(r'(?im)^(jump to content|cookie|privacy|sign in|topics|back|compliance|risk management|consumer banking|accessibility|share|print|subscribe|related|resources|tags|categories|contact|legal|terms|sitemap|search|hamburger menu)\b')
BULLET_NAV = re.compile(r'^\s*[-*+]\s*(?:\[[^\]]*\]\([^)]+\)|[A-Z][\w&\s]{0,40}|share|print|subscribe)\s*$', re.M)

FRAUD_TERMS = [
    "fraud","fraudulent","scam","embezzl","money laundering","bribery","corruption",
    "kickback","false claim","wire fraud","mail fraud","bank fraud","securities",
    "identity theft","phish","insider trading","fcpa","ponzi","ransomware","sanction",
    "enforcement","consent decree","scienter","mens rea"
]

END_MARKERS = re.compile(r'(?im)^\s*(##\s+In Depth|Press Contact|##\s+Resources for the Media|###\s*Sitemap|##\s*Privacy Preference Center|###\s*Cookie Settings|###\s*Connect With Us|©\s*\d{4}\s+American Bankers Association)\b')

def markdown_to_text(md: str) -> str:
    t = md
    t = MD_FRONTMATTER.sub('\n', t)
    t = MD_CODEBLOCK.sub('\n', t)
    t = MD_TABLE_LINE.sub('\n', t)
    t = MD_IMAGE.sub(' ', t)
    t = MD_LINK.sub(r'\1', t)
    t = MD_INLINECODE.sub(' ', t)
    t = MD_BLOCKQUOTE.sub('', t)
    t = MD_HEADINGS.sub('', t)
    t = MD_BOLD_ITALIC.sub(r'\1', t)
    t = unicodedata.normalize("NFC", t).translate(ZERO_WIDTH)
    lines = [ln.strip() for ln in t.splitlines()]
    kept = []
    for ln in lines:
        if not ln:
            kept.append('')
            continue
        if BANNED_LINE.match(ln):
            continue
        if BULLET_NAV.match(ln):
            continue
        if len(ln) < 4 and ln in {'*','-','•','·'}:
            continue
        kept.append(ln)
    t = "\n".join(kept)
    t = URL_RE.sub('', t)
    t = EMAIL_RE.sub('', t)
    t = EMOJI_RE.sub('', t)
    t = BRACKETS.sub('', t)
    t = re.sub(r'\n{2,}', '\n\n', t)
    t = re.sub(r'[ \t]+', ' ', t).strip()
    t = cut_article_core(t)
    return t.strip()

def cut_article_core(text: str) -> str:
    start = re.search(r'(?im)^\s*#\s+.+', text)
    if not start:
        start = re.search(r'(?im)^\s*Press Release\b.*\n+.*\n+^#\s+.+', text, re.M)
    if start:
        text = text[start.start():]
    end = END_MARKERS.search(text)
    if end:
        text = text[:end.start()]
    return text.strip()

def sent_split(text: str):
    return re.split(r'(?<=[\.\?\!])\s+(?=[A-Z(0-9])', text)

def relevant_sentences(text: str, window:int=1) -> str:
    sents = sent_split(text)
    keep = set()
    low_sents = [s.lower() for s in sents]
    for i, s in enumerate(low_sents):
        if any(term in s for term in FRAUD_TERMS):
            for j in range(max(0, i-window), min(len(sents), i+window+1)):
                keep.add(j)
    return " ".join(sents[i] for i in sorted(keep)).strip()

def fp(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def process_md(path: Path, debug=False):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    norm = markdown_to_text(raw)
    if not norm or len(norm) < 120:
        if debug: print(f"[SKIP {path}] empty/short", file=sys.stderr)
        return None
    rel = relevant_sentences(norm, window=1)
    use_text = rel if len(rel) >= 200 else norm
    return {
        "source_path": str(path),
        "title": path.stem,
        "clean_text": norm,
        "fraud_text": rel,
        "fingerprint": fp(use_text),
        "num_chars": len(use_text)
    }

def run(in_dir: str, out_path: str, limit=None, debug=False):
    base = Path(in_dir)
    files = sorted(base.rglob("*.md"))
    seen, kept = set(), 0
    total = len(files)
    if limit: files = files[:limit]
    if debug: print(f"[INFO] Found {total} .md under {in_dir}. Processing {len(files)} now.", file=sys.stderr)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for f in files:
            item = process_md(f, debug=debug)
            if not item: continue
            if item["fingerprint"] in seen: continue
            seen.add(item["fingerprint"])
            w.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    if debug: print(f"[DONE] Kept {kept}. Wrote -> {out_path}", file=sys.stderr)
    print(f"Wrote {kept} cleaned docs to {out_path} (seen {total} .md files)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/scraped_results")
    ap.add_argument("--out", default="data/clean/aba_clean.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    run(args.in_dir, args.out, args.limit or None, args.debug)
