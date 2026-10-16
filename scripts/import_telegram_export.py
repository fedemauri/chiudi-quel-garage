#!/usr/bin/env python3
"""Import garage photos from a Telegram Desktop chat export into a training dataset.

The Bot API cannot read chat history, so photos are taken from a manual export:
Telegram Desktop -> bot chat -> menu -> "Export chat history" -> Photos only.
Both export formats are supported: JSON (result.json) and HTML (messages*.html),
each with a photos/ directory.

Each photo is labeled from its caption (the label is what the model decided at
the time, NOT ground truth) and copied to:

    <out>/open/        status change to open, night alert, reminders, final warning
    <out>/closed/      status change to closed
    <out>/unlabeled/   /foto live snapshots and unrecognized captions

A manifest.csv lists every photo with heuristic flags that point at labels
worth reviewing (short closed spells, low confidence, reminders sent while closed).
Its review_label / review_note columns are left empty for manual review.

Usage:
    python scripts/import_telegram_export.py ~/Downloads/Telegram\\ Desktop/ChatExport_2026-10-16
    python scripts/import_telegram_export.py <export_dir> --out data/telegram --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data/telegram"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Caption kind -> label. Kinds map to the notifier methods that send photos.
KIND_LABELS = {
    "status_open": "open",
    "status_closed": "closed",
    "night_open": "open",
    "final_warning": "open",
    "reminder": "open",
    "live": None,
    "unknown": None,
}
# Status-change photos are the most informative (and the only source of
# "closed" images), so they are never dropped as near-duplicates.
TRANSITION_KINDS = {"status_open", "status_closed", "night_open"}

MANIFEST_FIELDS = [
    "file", "label", "kind", "date", "message_id", "confidence", "episode",
    "flags", "near_dup_of", "sha256", "source", "review_label", "review_note",
]

_CONFIDENCE_RE = re.compile(r"Confidenza:\s*(\d+)\s*%")

# HTML exports carry dates in the app language, e.g. "5 marzo 2026, 19:30:53"
# (Italian) or "05.03.2026 19:30:53 UTC+01:00" (English).
_MONTHS = {
    name: i + 1
    for names in (
        ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
         "agosto", "settembre", "ottobre", "novembre", "dicembre"),
        ("january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"),
    )
    for i, name in enumerate(names)
}
_DATE_WORDS_RE = re.compile(r"(\d{1,2}) (\w+) (\d{4}),? (\d{1,2}):(\d{2}):(\d{2})")
_DATE_DOTS_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4}) (\d{2}):(\d{2}):(\d{2})")


@dataclass
class PhotoRecord:
    message_id: int
    date: datetime
    source: Path
    caption: str
    kind: str = "unknown"
    label: str | None = None
    confidence: float | None = None
    episode: int | None = None
    flags: list[str] = field(default_factory=list)
    file: str = ""
    near_dup_of: str = ""
    sha256: str = ""


def message_text(msg: dict) -> str:
    """Flatten a Telegram export message text (str or list of str/entity dicts)."""
    text = msg.get("text", "")
    if isinstance(text, str):
        return text
    return "".join(part if isinstance(part, str) else part.get("text", "") for part in text)


def classify_caption(caption: str) -> str:
    """Return the caption kind for a photo sent by the garage monitor."""
    if "Stato cambiato" in caption and "→" in caption:
        # Text after the arrow is the new status; the old status precedes it
        # (Italian since March 2026, English before).
        new_status = caption.rsplit("→", 1)[1].split("\n", 1)[0].upper()
        if "APERTO" in new_status:
            return "status_open"
        if "CHIUSO" in new_status:
            return "status_closed"
        return "unknown"
    if "ATTENZIONE NOTTURNA" in caption:
        return "night_open"
    if "ULTIMO AVVISO" in caption:
        return "final_warning"
    if "ancora aperto da" in caption:
        return "reminder"
    if "Foto dal vivo" in caption:
        return "live"
    return "unknown"


def parse_confidence(caption: str) -> float | None:
    match = _CONFIDENCE_RE.search(caption)
    return int(match.group(1)) / 100 if match else None


def parse_html_date(title: str) -> datetime | None:
    if match := _DATE_WORDS_RE.match(title):
        day, month_name, year, hh, mm, ss = match.groups()
        month = _MONTHS.get(month_name.lower())
        if month is None:
            return None
        return datetime(int(year), month, int(day), int(hh), int(mm), int(ss))
    if match := _DATE_DOTS_RE.match(title):
        day, month, year, hh, mm, ss = map(int, match.groups())
        return datetime(year, month, day, hh, mm, ss)
    return None


class _ExportHTMLParser(HTMLParser):
    """Extract messages from Telegram Desktop HTML export pages.

    Produces dicts shaped like JSON export messages (id, type, date, photo, text)
    so the rest of the pipeline is format-agnostic.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.messages: list[dict] = []
        self._msg: dict | None = None
        self._depth = 0
        self._msg_depth: int | None = None
        self._text_depth: int | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: v or "" for k, v in attrs}
        classes = attr.get("class", "").split()
        if tag == "br":
            if self._text_depth is not None:
                self._text_parts.append("\n")
        elif tag == "a":
            if self._msg is not None and "photo_wrap" in classes:
                self._msg["photo"] = attr.get("href", "")
        elif tag == "div":
            self._depth += 1
            msg_id = attr.get("id", "").removeprefix("message")
            if "message" in classes and "service" not in classes and msg_id.isdigit():
                self._msg = {"id": int(msg_id), "type": "message", "text": ""}
                self._msg_depth = self._depth
            elif self._msg is not None:
                if "date" in classes and "title" in attr and "date" not in self._msg:
                    date = parse_html_date(attr["title"])
                    if date is not None:
                        self._msg["date"] = date.isoformat()
                elif "text" in classes and self._text_depth is None:
                    self._text_depth = self._depth
                    self._text_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "div":
            return
        if self._text_depth == self._depth and self._msg is not None:
            # Source line breaks are formatting only; real ones come from <br>.
            self._msg["text"] = "".join(self._text_parts).strip()
            self._text_depth = None
        if self._msg_depth == self._depth and self._msg is not None:
            if "date" in self._msg:
                self.messages.append(self._msg)
            self._msg = None
            self._msg_depth = None
        self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._text_depth is not None:
            self._text_parts.append(data.replace("\n", ""))


def _html_page_number(path: Path) -> int:
    suffix = path.stem.removeprefix("messages")
    return int(suffix) if suffix.isdigit() else 1


def load_html_messages(pages: list[Path]) -> list[dict]:
    parser = _ExportHTMLParser()
    for page in sorted(pages, key=_html_page_number):
        parser.feed(page.read_text(encoding="utf-8"))
    parser.close()
    return parser.messages


def load_messages(export_dir: Path, chat_name: str | None) -> list[dict]:
    """Load messages from a JSON (single-chat or full-account) or HTML export."""
    result_path = export_dir / "result.json"
    if not result_path.exists():
        pages = list(export_dir.glob("messages*.html"))
        if not pages:
            sys.exit(f"Neither result.json nor messages*.html found in {export_dir}")
        return load_html_messages(pages)
    data = json.loads(result_path.read_text(encoding="utf-8"))

    if "messages" in data:
        return data["messages"]

    chats = data.get("chats", {}).get("list", [])
    if chat_name is None:
        names = ", ".join(repr(c.get("name")) for c in chats)
        sys.exit(f"Full-account export: pass --chat with one of: {names}")
    for chat in chats:
        if chat.get("name") == chat_name:
            return chat.get("messages", [])
    sys.exit(f"Chat {chat_name!r} not found in export")


def _message_date(msg: dict) -> datetime:
    if "date_unixtime" in msg:
        return datetime.fromtimestamp(int(msg["date_unixtime"]))
    return datetime.fromisoformat(msg["date"])


def collect_photos(messages: list[dict], export_dir: Path) -> tuple[list[PhotoRecord], int]:
    """Build photo records sorted chronologically. Returns (records, missing_files)."""
    root = export_dir.resolve()
    records = []
    missing = 0
    for msg in messages:
        photo = msg.get("photo")
        if msg.get("type") != "message" or not photo:
            continue
        source = (export_dir / photo).resolve()
        # Paths come from the export file: never read outside the export folder.
        if not source.is_relative_to(root) or not source.is_file():
            missing += 1
            continue
        if source.suffix.lower() not in IMAGE_EXTS:
            continue
        caption = message_text(msg)
        kind = classify_caption(caption)
        records.append(PhotoRecord(
            message_id=int(msg["id"]),
            date=_message_date(msg),
            source=source,
            caption=caption,
            kind=kind,
            label=KIND_LABELS[kind],
            confidence=parse_confidence(caption),
        ))
    records.sort(key=lambda r: (r.date, r.message_id))
    return records, missing


def annotate(records: list[PhotoRecord], flip_minutes: float, low_confidence: float) -> None:
    """Assign open episodes and add review flags in place (records are chronological)."""
    episode = 0
    status: str | None = None
    prev_transition: PhotoRecord | None = None

    for rec in records:
        if rec.kind in TRANSITION_KINDS:
            if rec.label == "open":
                episode += 1
            # A "closed" that reopens within minutes is suspicious: one of the
            # two photos is likely misclassified. Short *open* spells are normal
            # (drive out, close at the next check) and are not flagged.
            if (
                prev_transition is not None
                and prev_transition.label == "closed"
                and rec.label == "open"
                and (rec.date - prev_transition.date).total_seconds() <= flip_minutes * 60
            ):
                for r in (prev_transition, rec):
                    if "short_closed" not in r.flags:
                        r.flags.append("short_closed")
            prev_transition = rec
            status = rec.label

        if rec.label == "open":
            rec.episode = episode
        if rec.kind in ("reminder", "final_warning") and status == "closed":
            rec.flags.append("reminder_while_closed")
        if rec.confidence is not None and rec.confidence < low_confidence:
            rec.flags.append("low_confidence")
        if rec.label is None:
            rec.flags.append("unlabeled")


def dhash(path: Path, size: int = 8) -> int:
    """64-bit difference hash, robust to Telegram recompression."""
    from PIL import Image

    with Image.open(path) as img:
        gray = img.convert("L").resize((size + 1, size))
        px = gray.tobytes()  # mode L: one byte per pixel
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def deduplicate(records: list[PhotoRecord], threshold: int) -> list[PhotoRecord]:
    """Drop exact duplicates and near-duplicate non-transition photos.

    A photo is a near-duplicate when its dHash is within `threshold` bits of
    the last kept photo with the same label (e.g. consecutive reminders of an
    unchanged open door). Returns the kept records; dropped ones get near_dup_of.
    """
    seen_sha: dict[str, PhotoRecord] = {}
    last_hash: dict[str | None, tuple[int, PhotoRecord]] = {}
    kept = []
    for rec in records:
        rec.sha256 = hashlib.sha256(rec.source.read_bytes()).hexdigest()
        if rec.sha256 in seen_sha:
            rec.near_dup_of = f"msg:{seen_sha[rec.sha256].message_id}"
            rec.flags.append("exact_duplicate")
            continue
        seen_sha[rec.sha256] = rec

        h = dhash(rec.source)
        prev = last_hash.get(rec.label)
        if (
            threshold > 0
            and rec.kind not in TRANSITION_KINDS
            and prev is not None
            and bin(h ^ prev[0]).count("1") <= threshold
        ):
            rec.near_dup_of = f"msg:{prev[1].message_id}"
            rec.flags.append("near_duplicate")
            continue
        last_hash[rec.label] = (h, rec)
        kept.append(rec)
    return kept


def write_dataset(records: list[PhotoRecord], kept: list[PhotoRecord], out: Path) -> None:
    for rec in kept:
        folder = rec.label or "unlabeled"
        name = f"{rec.date:%Y%m%d_%H%M%S}_{rec.message_id}{rec.source.suffix.lower()}"
        rec.file = f"{folder}/{name}"
        dest = out / rec.file
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec.source, dest)

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "file": rec.file,
                "label": rec.label or "",
                "kind": rec.kind,
                "date": rec.date.isoformat(timespec="seconds"),
                "message_id": rec.message_id,
                "confidence": "" if rec.confidence is None else f"{rec.confidence:.2f}",
                "episode": rec.episode or "",
                "flags": ";".join(rec.flags),
                "near_dup_of": rec.near_dup_of,
                "sha256": rec.sha256,
                "source": rec.source.name,
                "review_label": "",
                "review_note": "",
            })


def print_summary(records: list[PhotoRecord], kept: list[PhotoRecord], missing: int) -> None:
    print(f"Photos in export: {len(records)}" + (f" ({missing} missing files skipped)" if missing else ""))
    if records:
        print(f"Period: {records[0].date:%Y-%m-%d} -> {records[-1].date:%Y-%m-%d}")
    print("\nBy caption kind:")
    for kind, n in Counter(r.kind for r in records).most_common():
        print(f"  {kind:<15} {n}")
    print(f"\nKept after dedup: {len(kept)}")
    for label, n in Counter(r.label or "unlabeled" for r in kept).most_common():
        print(f"  {label:<15} {n}")
    flags = Counter(f for r in records for f in r.flags)
    if flags:
        print("\nFlags (all photos):")
        for flag, n in flags.most_common():
            print(f"  {flag:<22} {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", type=Path, help="Telegram Desktop export folder (contains result.json)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output dataset folder (default: {DEFAULT_OUT})")
    parser.add_argument("--chat", help="Chat name, only for full-account exports")
    parser.add_argument("--flip-minutes", type=float, default=10, help="Flag closed spells shorter than this (default: 10)")
    parser.add_argument("--low-confidence", type=float, default=0.85, help="Flag captions below this confidence (default: 0.85)")
    parser.add_argument("--near-dup-threshold", type=int, default=4,
                        help="Max dHash bit distance to drop non-transition near-duplicates; 0 disables (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Print the summary without copying files")
    parser.add_argument("--force", action="store_true", help="Write into a non-empty output folder")
    args = parser.parse_args()

    if not args.dry_run and args.out.exists() and any(args.out.iterdir()) and not args.force:
        sys.exit(f"{args.out} is not empty: use --force or choose another --out")

    messages = load_messages(args.export_dir, args.chat)
    records, missing = collect_photos(messages, args.export_dir)
    if not records:
        sys.exit("No photos found in export (did you include Photos when exporting?)")

    annotate(records, args.flip_minutes, args.low_confidence)
    kept = deduplicate(records, args.near_dup_threshold)
    print_summary(records, kept, missing)

    if args.dry_run:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    write_dataset(records, kept, args.out)
    print(f"\nDataset written to {args.out} (manifest: {args.out / 'manifest.csv'})")


if __name__ == "__main__":
    main()
