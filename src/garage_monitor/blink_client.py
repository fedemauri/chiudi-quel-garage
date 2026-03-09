"""Blink camera client: authentication with saved credentials and snapshot capture.

Uses the blinkpy library to connect to Blink cameras without interactive 2FA,
leveraging credentials previously saved to Firestore during initial setup.
"""

from __future__ import annotations

import asyncio
import logging

from aiohttp import ClientSession
from blinkpy.auth import Auth
from blinkpy.blinkpy import Blink

logger = logging.getLogger(__name__)


class BlinkAuthExpiredError(Exception):
    """Raised when Blink authentication has expired and 2FA is required."""

    pass


class BlinkClient:
    """Handles Blink authentication and snapshot capture.

    Typical usage:
        client = BlinkClient()
        await client.connect(credentials_from_firestore)
        jpeg_bytes = await client.take_snapshot("Garage")
        updated_creds = client.get_updated_credentials()
        # save updated_creds back to Firestore
        await client.close()
    """

    def __init__(self) -> None:
        self._blink: Blink | None = None
        self._session: ClientSession | None = None

    async def connect(self, credentials: dict) -> None:
        """Authenticate to Blink using saved credentials (no 2FA needed).

        Args:
            credentials: Dict with login_attributes previously obtained
                from an interactive 2FA setup and saved to Firestore.

        Raises:
            BlinkAuthExpiredError: If saved tokens are expired and a new
                interactive 2FA setup is required.
        """
        try:
            self._session = ClientSession()
            auth = Auth(credentials, no_prompt=True)
            self._blink = Blink(session=self._session)
            self._blink.auth = auth
            await self._blink.start()
            logger.info(
                "Blink connected. Cameras found: %s",
                list(self._blink.cameras.keys()),
            )
        except Exception as e:
            logger.error("Blink auth failed: %s", e)
            await self.close()
            raise BlinkAuthExpiredError(
                "Blink authentication expired. Run setup_blink.py again."
            ) from e

    async def take_snapshot(self, camera_name: str) -> bytes:
        """Take a fresh snapshot and return JPEG bytes.

        Triggers a new snapshot on the camera, refreshes the Blink state
        to download it, then returns the cached image bytes.

        Args:
            camera_name: Name of the Blink camera (e.g. "Garage").

        Returns:
            JPEG image bytes.

        Raises:
            RuntimeError: If connect() has not been called.
            ValueError: If camera_name is not found.
            RuntimeError: If snapshot bytes are empty after capture.
        """
        if not self._blink:
            raise RuntimeError("Not connected. Call connect() first.")

        camera = self._blink.cameras.get(camera_name)
        if not camera:
            available = list(self._blink.cameras.keys())
            raise ValueError(
                f"Camera '{camera_name}' not found. Available: {available}"
            )

        logger.info("Requesting snapshot from camera '%s'...", camera_name)
        await camera.snap_picture()
        await asyncio.sleep(3)
        await self._blink.refresh(force=True)

        image_bytes: bytes = camera.image_from_cache
        if not image_bytes:
            raise RuntimeError(
                f"No image data received from camera '{camera_name}'."
            )

        logger.info(
            "Snapshot captured from '%s' (%d bytes).",
            camera_name,
            len(image_bytes),
        )
        return image_bytes

    def get_updated_credentials(self) -> dict:
        """Return refreshed credentials to save back to Firestore.

        After a successful connect + API calls, the auth tokens may have
        been refreshed. Call this to get the updated login_attributes dict.

        Returns:
            Dict with current login attributes (tokens, account info, etc.).

        Raises:
            RuntimeError: If connect() has not been called.
        """
        if not self._blink:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._blink.auth.login_attributes

    async def close(self) -> None:
        """Close the aiohttp session and release resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._blink = None
