"""take_fresh_snapshot: the photo must really be new (the Blink thumbnail ts must change after the snap)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from garage_monitor.blink_client import BlinkClient, StalePhotoError, thumbnail_ts

OLD = "https://rest-e007.immedia-semi.com/api/v3/media/accounts/1/networks/2/owl/3/thumbnail/thumbnail.jpg?ts=1795006505&ext="
NEW = OLD.replace("1795006505", "1795007076")


def _client(thumbnails):
    """A BlinkClient whose camera shows OLD, then each refresh moves to the next thumbnail."""
    camera = MagicMock()
    camera.thumbnail = OLD
    camera.image_from_cache = b"\xff\xd8old"
    camera.snap_picture = AsyncMock()
    seq = iter(thumbnails)

    async def refresh(force=False):
        url = next(seq)
        camera.thumbnail = url
        camera.image_from_cache = b"\xff\xd8new" if url != OLD else b"\xff\xd8old"

    blink = MagicMock()
    blink.cameras = {"Garage": camera}
    blink.refresh = AsyncMock(side_effect=refresh)
    client = BlinkClient()
    client._blink = blink
    return client, camera, blink


def test_thumbnail_ts():
    assert thumbnail_ts(OLD) == 1795006505
    assert thumbnail_ts(None) is None
    assert thumbnail_ts("https://x/thumbnail.jpg?ext=") is None


@patch("garage_monitor.blink_client.asyncio.sleep", new_callable=AsyncMock)
def test_fresh_photo_first_try(sleep):
    client, camera, blink = _client([NEW])
    assert asyncio.run(client.take_fresh_snapshot("Garage")) == (b"\xff\xd8new", 1795007076)
    camera.snap_picture.assert_awaited_once()
    blink.refresh.assert_awaited_once_with(force=True)
    sleep.assert_awaited_once_with(3)


@patch("garage_monitor.blink_client.asyncio.sleep", new_callable=AsyncMock)
def test_fresh_photo_after_retries(sleep):
    client, _, blink = _client([OLD, OLD, NEW])
    assert asyncio.run(client.take_fresh_snapshot("Garage")) == (b"\xff\xd8new", 1795007076)
    assert blink.refresh.await_count == 3


@patch("garage_monitor.blink_client.asyncio.sleep", new_callable=AsyncMock)
def test_stale_photo(sleep):
    client, _, blink = _client([OLD, OLD, OLD])
    with pytest.raises(StalePhotoError):
        asyncio.run(client.take_fresh_snapshot("Garage"))
    assert blink.refresh.await_count == 3


@patch("garage_monitor.blink_client.asyncio.sleep", new_callable=AsyncMock)
def test_unknown_camera(sleep):
    client, _, _ = _client([NEW])
    with pytest.raises(ValueError):
        asyncio.run(client.take_fresh_snapshot("Cucina"))
