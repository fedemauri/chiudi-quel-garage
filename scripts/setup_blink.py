#!/usr/bin/env python3
"""Setup interattivo per credenziali Blink con 2FA.

Eseguire una sola volta in locale. Salva le credenziali su Firestore
per l'uso da parte della Cloud Function.
"""

import asyncio
import os
import sys

from aiohttp import ClientSession
from blinkpy.auth import Auth
from blinkpy.blinkpy import Blink
from blinkpy.auth import BlinkTwoFARequiredError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from garage_monitor.firestore_store import FirestoreStore


async def main():
    project_id = os.environ.get("GM_GCP_PROJECT_ID")
    collection = os.environ.get("GM_FIRESTORE_COLLECTION", "garage_monitor")

    if not project_id:
        project_id = input("GCP Project ID: ").strip()

    email = os.environ.get("GM_BLINK_USERNAME") or input("Email Blink: ").strip()
    password = os.environ.get("GM_BLINK_PASSWORD") or input("Password Blink: ").strip()
    camera_name = os.environ.get("GM_BLINK_CAMERA_NAME", "Garage")

    print("\n>>> Autenticazione Blink...")
    session = ClientSession()
    try:
        auth = Auth({"username": email, "password": password}, no_prompt=True)
        blink = Blink(session=session)
        blink.auth = auth
        try:
            await blink.start()
        except BlinkTwoFARequiredError:
            print("\n2FA richiesto! Controlla la tua email per il codice PIN.")
            pin = input("Inserisci il PIN 2FA: ").strip()
            await blink.send_2fa_code(pin)

        print(f"\nCamere trovate: {list(blink.cameras.keys())}")

        camera = blink.cameras.get(camera_name)
        if not camera:
            print(f"ATTENZIONE: Camera '{camera_name}' non trovata!")
            print(f"Camere disponibili: {list(blink.cameras.keys())}")
        else:
            print(f"\n>>> Test snapshot dalla camera '{camera_name}'...")
            await camera.snap_picture()
            await blink.refresh(force=True)
            image = camera.image_from_cache
            if image:
                test_path = "/tmp/garage_test.jpg"
                with open(test_path, "wb") as f:
                    f.write(image)
                print(f"Snapshot salvato in {test_path} ({len(image)} bytes)")
            else:
                print("ATTENZIONE: Snapshot vuoto!")

        credentials = blink.auth.login_attributes
        print("\n>>> Salvataggio credenziali su Firestore...")
        store = FirestoreStore(project_id, collection)
        store.save_blink_credentials(credentials)
        print("Credenziali salvate con successo!")

        print("\n=== Setup completato! ===")
        print("La Cloud Function potra' ora autenticarsi automaticamente.")
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
