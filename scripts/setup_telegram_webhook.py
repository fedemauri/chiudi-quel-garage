#!/usr/bin/env python3
"""Register or delete a Telegram webhook for the Cloud Function.

Usage:
    source .env
    python scripts/setup_telegram_webhook.py <FUNCTION_URL>
    python scripts/setup_telegram_webhook.py --delete
"""

import os
import sys

import httpx

BOT_TOKEN = os.environ.get("GM_TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET = os.environ.get("GM_TELEGRAM_WEBHOOK_SECRET", "")


def main():
    if not BOT_TOKEN:
        print("ERROR: GM_TELEGRAM_BOT_TOKEN not set. Run: source .env")
        sys.exit(1)

    base_url = f"https://api.telegram.org/bot{BOT_TOKEN}"

    if "--delete" in sys.argv:
        resp = httpx.post(f"{base_url}/deleteWebhook")
        resp.raise_for_status()
        print("Webhook deleted:", resp.json())
        return

    if len(sys.argv) < 2:
        print("Usage: python scripts/setup_telegram_webhook.py <FUNCTION_URL>")
        print("       python scripts/setup_telegram_webhook.py --delete")
        sys.exit(1)

    function_url = sys.argv[1]

    if not WEBHOOK_SECRET:
        print("WARNING: GM_TELEGRAM_WEBHOOK_SECRET not set. Webhook will not be secured.")

    params = {"url": function_url}
    if WEBHOOK_SECRET:
        params["secret_token"] = WEBHOOK_SECRET

    resp = httpx.post(f"{base_url}/setWebhook", json=params)
    resp.raise_for_status()
    print("Webhook set:", resp.json())

    # Verify
    resp = httpx.get(f"{base_url}/getWebhookInfo")
    resp.raise_for_status()
    info = resp.json().get("result", {})
    print(f"URL: {info.get('url')}")
    print(f"Has custom certificate: {info.get('has_custom_certificate')}")
    print(f"Pending update count: {info.get('pending_update_count')}")


if __name__ == "__main__":
    main()
