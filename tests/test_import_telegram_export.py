import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts/import_telegram_export.py"
_spec = importlib.util.spec_from_file_location("import_telegram_export", _SCRIPT)
ite = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ite  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(ite)


class TestClassifyCaption:
    @pytest.mark.parametrize(
        "caption,kind",
        [
            ("🏠 Garage Monitor\n\nStato cambiato: chiuso → 🔴 APERTO\nConfidenza: 95%\nPorta su", "status_open"),
            ("🏠 Garage Monitor\n\nStato cambiato: aperto → 🟢 CHIUSO\nConfidenza: 90%\nPorta giù", "status_closed"),
            # Pre-March 2026 format: English old status, "Motivo:"
            ("Stato cambiato: open → 🟢 CHIUSO\nConfidenza: 90%\nMotivo: la porta è aperta?", "status_closed"),
            ("🚨🚨🚨 ATTENZIONE NOTTURNA 🚨🚨🚨\n\nIl garage si e' APERTO di notte!", "night_open"),
            ("⚠️ ULTIMO AVVISO — Il box e' aperto da 75 minuti!", "final_warning"),
            ("🏠 Garage Monitor\n\n⏰ Il box e' ancora aperto da 15 minuti!", "reminder"),
            ("📷 Foto dal vivo", "live"),
            ("", "unknown"),
        ],
    )
    def test_kinds(self, caption, kind):
        assert ite.classify_caption(caption) == kind

    def test_reasoning_words_do_not_override_new_status(self):
        caption = "Stato cambiato: aperto → 🟢 CHIUSO\nConfidenza: 90%\nNon più APERTO"
        assert ite.classify_caption(caption) == "status_closed"


def test_message_text_flattens_entities():
    msg = {"text": ["🏠 ", {"type": "bold", "text": "Garage Monitor"}, "\n\nok"]}
    assert ite.message_text(msg) == "🏠 Garage Monitor\n\nok"


def test_parse_confidence():
    assert ite.parse_confidence("Confidenza: 87%") == pytest.approx(0.87)
    assert ite.parse_confidence("📷 Foto dal vivo") is None


def _write_image(path: Path, seed: int, quality: int = 75) -> None:
    """Write a distinct image per seed (different gradients -> different dHash)."""
    img = Image.new("L", (90, 80))
    img.putdata([((x * (seed + 1) + y * seed * 7) % 256) for y in range(80) for x in range(90)])
    img.convert("RGB").save(path, "JPEG", quality=quality)


@pytest.fixture
def export_dir(tmp_path):
    root = tmp_path / "ChatExport"
    (root / "photos").mkdir(parents=True)
    # (id, time, caption, image seed, jpeg quality); same seed + other quality =
    # same scene recompressed (near-duplicate, different bytes)
    rows = [
        (1, "2026-05-01T08:00:00", "Stato cambiato: chiuso → 🔴 APERTO\nConfidenza: 95%", 1, 75),
        (2, "2026-05-01T08:15:00", "⏰ Il box e' ancora aperto da 15 minuti!", 1, 60),
        (3, "2026-05-01T08:20:00", "Stato cambiato: aperto → 🟢 CHIUSO\nConfidenza: 72%", 2, 75),
        (4, "2026-05-01T09:00:00", "⏰ Il box e' ancora aperto da 60 minuti!", 3, 75),
        (5, "2026-05-01T10:00:00", "📷 Foto dal vivo", 4, 75),
        (8, "2026-05-01T10:05:00", "📷 Foto dal vivo", 4, 75),
        (9, "2026-05-01T10:10:00", "Stato cambiato: chiuso → 🔴 APERTO\nConfidenza: 90%", 5, 75),
    ]
    messages = []
    for msg_id, date, caption, seed, quality in rows:
        name = f"photos/photo_{msg_id}.jpg"
        _write_image(root / name, seed, quality)
        messages.append({"id": msg_id, "type": "message", "date": date, "photo": name, "text": caption})
    messages.append({"id": 6, "type": "message", "date": "2026-05-01T10:01:00", "text": "/foto"})
    messages.append({"id": 7, "type": "message", "date": "2026-05-01T10:02:00",
                     "photo": "../../etc/passwd", "text": "📷 Foto dal vivo"})
    (root / "result.json").write_text(json.dumps({"name": "Garage", "messages": messages}))
    return root


def test_end_to_end(export_dir, tmp_path):
    out = tmp_path / "out"
    records, missing = ite.collect_photos(ite.load_messages(export_dir, None), export_dir)
    assert missing == 1  # path outside the export is ignored
    assert [r.message_id for r in records] == [1, 2, 3, 4, 5, 8, 9]

    ite.annotate(records, flip_minutes=120, low_confidence=0.85)
    kept = ite.deduplicate(records, threshold=4)
    out.mkdir()
    ite.write_dataset(records, kept, out)

    by_id = {r.message_id: r for r in records}
    # closed 08:20 -> open 10:10 is a short closed spell; open 08:00 -> closed 08:20 is normal
    assert "short_closed" in by_id[3].flags and "short_closed" in by_id[9].flags
    assert "short_closed" not in by_id[1].flags
    assert "low_confidence" in by_id[3].flags
    assert "reminder_while_closed" in by_id[4].flags
    assert by_id[2].near_dup_of == "msg:1"  # same picture as the opening photo
    assert "exact_duplicate" in by_id[8].flags and by_id[8].near_dup_of == "msg:5"
    assert by_id[1].episode == 1 and by_id[2].episode == 1

    assert sorted(p.name for p in (out / "open").iterdir()) == [
        "20260501_080000_1.jpg", "20260501_090000_4.jpg", "20260501_101000_9.jpg",
    ]
    assert [p.name for p in (out / "closed").iterdir()] == ["20260501_082000_3.jpg"]
    assert [p.name for p in (out / "unlabeled").iterdir()] == ["20260501_100000_5.jpg"]

    with open(out / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7
    assert rows[1]["file"] == "" and "near_duplicate" in rows[1]["flags"]


_HTML_PAGE = """<div class="history">
<div class="message service" id="message-1"><div class="body details">5 marzo 2026</div></div>
<div class="message default clearfix" id="message5">
<div class="pull_left userpic_wrap"><img class="userpic" src="photos/author_1.jpg"/></div>
<div class="body">
<div class="pull_right date details" title="5 marzo 2026, 20:50:17">20:50</div>
<div class="from_name">
chiudi-quel-garage
</div>
<a class="photo_wrap clearfix pull_left" href="photos/photo_5_0.jpg"><img class="photo" src="photos/photo_5_0.jpg"></a>
<div class="text">
🏠 <strong>Garage Monitor</strong><br><br>Stato cambiato: chiuso → <strong>🔴 APERTO</strong><br>Confidenza: 100%<br>Porta &amp; interno
</div>
</div>
</div>
<div class="message default clearfix joined" id="message7">
<div class="body">
<div class="pull_right date details" title="05.03.2026 21:00:01 UTC+01:00">21:00</div>
<div class="text">
📊 <strong>Report</strong>
</div>
</div>
</div>
</div>"""


def test_load_html_messages(tmp_path):
    (tmp_path / "messages.html").write_text(_HTML_PAGE, encoding="utf-8")
    messages = ite.load_messages(tmp_path, None)

    assert [m["id"] for m in messages] == [5, 7]
    first = messages[0]
    assert first["date"] == "2026-03-05T20:50:17"
    assert first["photo"] == "photos/photo_5_0.jpg"
    assert first["text"] == "🏠 Garage Monitor\n\nStato cambiato: chiuso → 🔴 APERTO\nConfidenza: 100%\nPorta & interno"
    assert ite.classify_caption(first["text"]) == "status_open"
    assert messages[1]["date"] == "2026-03-05T21:00:01" and "photo" not in messages[1]
