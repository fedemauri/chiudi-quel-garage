import logging
from datetime import datetime, timezone

import httpx

from garage_monitor.models import GarageState, UsageStats

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

    def send_night_alert(
        self,
        confidence: float,
        reasoning: str,
        photo_bytes: bytes | None = None,
    ) -> None:
        """Send high-priority alert when garage opens at night."""
        text = (
            "\U0001f6a8\U0001f6a8\U0001f6a8 *ATTENZIONE NOTTURNA* \U0001f6a8\U0001f6a8\U0001f6a8\n\n"
            "Il garage si e' APERTO di notte!\n"
            f"Confidenza: {confidence:.0%}\n"
            f"{reasoning}"
        )
        if photo_bytes:
            self._send_photo(photo_bytes, text)
        else:
            self._send_message(text)

    def send_final_warning(
        self, minutes_open: int, photo_bytes: bytes | None = None
    ) -> None:
        """Send final escalation warning after extended open time."""
        text = (
            "\u26a0\ufe0f *ULTIMO AVVISO* \u2014 "
            f"Il box e' aperto da {minutes_open} minuti!\n"
            "Non verranno inviate altre notifiche."
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

    def send_usage_report(
        self,
        stats: UsageStats,
        days_in_period: int,
        cost_warning: str | None = None,
    ) -> None:
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
        if cost_warning:
            warnings.append(cost_warning)

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
            f"*Attivita' garage*\n"
            f"  Aperture mese: {stats.garage_openings:,}\n\n"
            f"*Cloud Scheduler*: 3 / 3 job free\n\n"
            f"*TOTALE STIMATO*: ~${gemini_cost:.2f}\n"
            f"{'Tutto dentro il free tier ✅' if not warnings else ''}"
            f"{warning_text}"
        )
        self._send_message(text)

    def send_history(self, events: list[dict]) -> None:
        """Send formatted event history."""
        if not events:
            self._send_message("📋 Nessun evento registrato.")
            return

        lines = ["📋 *Storico eventi*\n"]
        for ev in events:
            ts = ev["timestamp"]
            date_str = ts.strftime("%d/%m %H:%M")
            old_s = self._STATUS_IT.get(ev.get("old_status", ""), ev.get("old_status", "?"))
            new_s = self._STATUS_IT.get(ev.get("new_status", ""), ev.get("new_status", "?"))
            emoji = "🔴" if ev.get("new_status") == "open" else "🟢"
            line = f"{emoji} {date_str} — {old_s} → {new_s}"
            duration = ev.get("duration_seconds")
            if duration is not None and ev.get("new_status") == "closed":
                minutes = duration // 60
                line += f" (aperto {minutes} min)"
            lines.append(line)

        self._send_message("\n".join(lines))

    def send_command_response(self, text: str) -> None:
        """Send a simple text response to a bot command."""
        self._send_message(text)

    def send_current_status(self, state: GarageState) -> None:
        """Send formatted current status for /stato command."""
        status_it = self._STATUS_IT.get(state.current_status.value, state.current_status.value)
        lines = [
            "\U0001f3e0 *Garage Monitor - Stato*\n",
            f"Stato attuale: *{status_it.upper()}*",
        ]
        if state.last_change_time:
            now = datetime.now(timezone.utc)
            delta = now - state.last_change_time
            minutes = int(delta.total_seconds() / 60)
            if minutes < 60:
                lines.append(f"Dall'ultimo cambio: {minutes} min")
            else:
                hours = minutes // 60
                mins = minutes % 60
                lines.append(f"Dall'ultimo cambio: {hours}h {mins}min")
        if state.muted_until:
            now = datetime.now(timezone.utc)
            if state.muted_until > now:
                remaining = int((state.muted_until - now).total_seconds() / 60)
                lines.append(f"\U0001f507 Notifiche silenziate per ancora {remaining} min")
        self._send_message("\n".join(lines))

    def send_photo(self, photo_bytes: bytes, caption: str = "") -> None:
        """Send a photo (public wrapper for bot commands)."""
        self._send_photo(photo_bytes, caption or "\U0001f4f7 Foto dal vivo")

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
