import logging
import httpx

from garage_monitor.models import UsageStats

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"

_MONTHS = [
    "", "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
]


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self._token = bot_token
        self._chat_id = chat_id
        self._base_url = TELEGRAM_API.format(token=bot_token)

    _STATUS_IT = {"open": "aperto", "closed": "chiuso", "unknown": "sconosciuto"}

    def send_status_change(
        self,
        old_status: str,
        new_status: str,
        confidence: float,
        reasoning: str,
        photo_bytes: bytes | None = None,
    ) -> None:
        """Send notification when garage status changes, optionally with photo."""
        old_it = self._STATUS_IT.get(old_status, old_status)
        status_emoji = "🔴 APERTO" if new_status == "open" else "🟢 CHIUSO"
        text = (
            f"🏠 *Garage Monitor*\n\n"
            f"Stato cambiato: {old_it} → *{status_emoji}*\n"
            f"Confidenza: {confidence:.0%}\n"
            f"{reasoning}"
        )

        if photo_bytes:
            self._send_photo(photo_bytes, text)
        else:
            self._send_message(text)

    def send_error_alert(self, message: str) -> None:
        """Send error alert after consecutive failures."""
        text = f"⚠️ *Garage Monitor - Errore*\n\n{message}"
        self._send_message(text)

    def send_monitor_started(self) -> None:
        """Send notification when monitor starts for the first time."""
        text = "🏠 *Garage Monitor*\n\n✅ Monitor avviato con successo!"
        self._send_message(text)

    def send_still_open_reminder(
        self, minutes_open: int, photo_bytes: bytes | None = None
    ) -> None:
        """Send reminder that garage is still open."""
        text = (
            f"🏠 *Garage Monitor*\n\n"
            f"⏰ Il box e' ancora aperto da {minutes_open} minuti!"
        )
        if photo_bytes:
            self._send_photo(photo_bytes, text)
        else:
            self._send_message(text)

    def send_usage_report(self, stats: UsageStats, days_in_period: int) -> None:
        """Send usage report with free tier comparison."""
        days = max(days_in_period, 1)
        fn_pct = stats.function_invocations / 2_000_000 * 100
        reads_daily = stats.firestore_reads / days
        writes_daily = stats.firestore_writes / days
        reads_pct = reads_daily / 50_000 * 100
        writes_pct = writes_daily / 20_000 * 100
        gemini_cost = (
            stats.gemini_input_tokens * 0.30
            + stats.gemini_output_tokens * 2.50
        ) / 1_000_000

        warnings = []
        if fn_pct > 80:
            warnings.append("Cloud Function invocazioni > 80% free tier!")
        if reads_pct > 80:
            warnings.append("Firestore letture giornaliere > 80% free tier!")
        if writes_pct > 80:
            warnings.append("Firestore scritture giornaliere > 80% free tier!")

        try:
            year, month = stats.period.split("_")
            period_name = f"{_MONTHS[int(month)]} {year}"
        except (ValueError, IndexError):
            period_name = stats.period

        warning_text = ""
        if warnings:
            warning_text = "\n⚠️ " + "\n⚠️ ".join(warnings)

        text = (
            f"📊 *Garage Monitor - Utilizzo Risorse*\n"
            f"Periodo: {period_name} ({days_in_period} giorni)\n\n"
            f"*Cloud Function*\n"
            f"  Invocazioni: {stats.function_invocations:,} / 2.000.000 free ({fn_pct:.1f}%)\n\n"
            f"*Gemini API*\n"
            f"  Chiamate: {stats.gemini_calls:,}\n"
            f"  Token input: ~{stats.gemini_input_tokens:,}\n"
            f"  Token output: ~{stats.gemini_output_tokens:,}\n"
            f"  Costo stimato: ~${gemini_cost:.2f}\n\n"
            f"*Firestore*\n"
            f"  Letture mese: {stats.firestore_reads:,} (~{reads_daily:.0f}/g vs 50.000/g free)\n"
            f"  Scritture mese: {stats.firestore_writes:,} (~{writes_daily:.0f}/g vs 20.000/g free)\n\n"
            f"*Cloud Scheduler*: 2 / 3 job free\n\n"
            f"*TOTALE STIMATO*: ~${gemini_cost:.2f}\n"
            f"{'Tutto dentro il free tier ✅' if not warnings else ''}"
            f"{warning_text}"
        )
        self._send_message(text)

    def _send_message(self, text: str) -> None:
        with httpx.Client() as client:
            resp = client.post(
                f"{self._base_url}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
            )
            resp.raise_for_status()

    def _send_photo(self, photo_bytes: bytes, caption: str) -> None:
        with httpx.Client() as client:
            resp = client.post(
                f"{self._base_url}/sendPhoto",
                data={
                    "chat_id": self._chat_id,
                    "caption": caption,
                    "parse_mode": "Markdown",
                },
                files={"photo": ("garage.jpg", photo_bytes, "image/jpeg")},
            )
            resp.raise_for_status()
