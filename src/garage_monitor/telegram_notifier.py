import logging
import httpx

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self._token = bot_token
        self._chat_id = chat_id
        self._base_url = TELEGRAM_API.format(token=bot_token)

    def send_status_change(
        self,
        old_status: str,
        new_status: str,
        confidence: float,
        reasoning: str,
        photo_bytes: bytes | None = None,
    ) -> None:
        """Send notification when garage status changes, optionally with photo."""
        status_emoji = "🔴 APERTO" if new_status == "open" else "🟢 CHIUSO"
        text = (
            f"🏠 *Garage Monitor*\n\n"
            f"Stato cambiato: {old_status} → *{status_emoji}*\n"
            f"Confidenza: {confidence:.0%}\n"
            f"Motivo: {reasoning}"
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
