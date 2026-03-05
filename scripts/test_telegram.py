#!/usr/bin/env python3
"""Verifica che il bot Telegram sia configurato correttamente."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from garage_monitor.telegram_notifier import TelegramNotifier


def main():
    token = os.environ.get("GM_TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("GM_TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("ERROR: Imposta GM_TELEGRAM_BOT_TOKEN e GM_TELEGRAM_CHAT_ID")
        print("  export GM_TELEGRAM_BOT_TOKEN=your-token")
        print("  export GM_TELEGRAM_CHAT_ID=your-chat-id")
        sys.exit(1)

    notifier = TelegramNotifier(bot_token=token, chat_id=chat_id)

    print("Invio messaggio di test...")
    notifier.send_monitor_started()
    print("OK! Controlla Telegram.")


if __name__ == "__main__":
    main()
