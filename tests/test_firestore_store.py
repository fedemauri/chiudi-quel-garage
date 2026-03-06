from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from garage_monitor.firestore_store import FirestoreStore
from garage_monitor.models import GarageState, GarageStatus


class TestFirestoreStore:
    @patch("garage_monitor.firestore_store.FirestoreClient")
    def test_get_state_not_found(self, mock_client_cls):
        mock_db = MagicMock()
        mock_client_cls.return_value = mock_db
        snap = MagicMock()
        snap.exists = False
        mock_db.collection.return_value.document.return_value.get.return_value = snap

        store = FirestoreStore("test-project")
        assert store.get_state() is None

    @patch("garage_monitor.firestore_store.FirestoreClient")
    def test_get_state_found(self, mock_client_cls):
        mock_db = MagicMock()
        mock_client_cls.return_value = mock_db
        now = datetime.now(timezone.utc)
        snap = MagicMock()
        snap.exists = True
        snap.to_dict.return_value = {
            "current_status": "closed",
            "last_check_time": now,
            "last_change_time": now,
            "consecutive_errors": 0,
        }
        mock_db.collection.return_value.document.return_value.get.return_value = snap

        store = FirestoreStore("test-project")
        state = store.get_state()

        assert state is not None
        assert state.current_status == GarageStatus.CLOSED
        assert state.consecutive_errors == 0

    @patch("garage_monitor.firestore_store.FirestoreClient")
    def test_save_state(self, mock_client_cls):
        mock_db = MagicMock()
        mock_client_cls.return_value = mock_db
        now = datetime.now(timezone.utc)

        store = FirestoreStore("test-project")
        state = GarageState(
            current_status=GarageStatus.OPEN,
            last_check_time=now,
            last_change_time=now,
            consecutive_errors=1,
        )
        store.save_state(state)

        mock_db.collection.return_value.document.return_value.set.assert_called_once()
        call_data = (
            mock_db.collection.return_value.document.return_value.set.call_args[0][0]
        )
        assert call_data["current_status"] == "open"
        assert call_data["consecutive_errors"] == 1

    @patch("garage_monitor.firestore_store.FirestoreClient")
    def test_get_blink_credentials_not_found(self, mock_client_cls):
        mock_db = MagicMock()
        mock_client_cls.return_value = mock_db
        snap = MagicMock()
        snap.exists = False
        mock_db.collection.return_value.document.return_value.get.return_value = snap

        store = FirestoreStore("test-project")
        assert store.get_blink_credentials() is None

    @patch("garage_monitor.firestore_store.FirestoreClient")
    def test_save_blink_credentials(self, mock_client_cls):
        mock_db = MagicMock()
        mock_client_cls.return_value = mock_db

        store = FirestoreStore("test-project")
        store.save_blink_credentials({"token": "abc"})

        call_data = (
            mock_db.collection.return_value.document.return_value.set.call_args[0][0]
        )
        assert call_data["token"] == "abc"
        assert "updated_at" in call_data

    @patch("garage_monitor.firestore_store.FirestoreClient")
    def test_save_event(self, mock_client_cls):
        mock_db = MagicMock()
        mock_client_cls.return_value = mock_db
        now = datetime(2026, 3, 6, 14, 30, 0, tzinfo=timezone.utc)

        store = FirestoreStore("test-project")
        event = {
            "timestamp": now,
            "old_status": "closed",
            "new_status": "open",
            "confidence": 0.9,
            "reasoning": "Door raised",
            "duration_seconds": None,
            "expire_at": now + timedelta(days=30),
        }
        store.save_event(event)

        mock_db.collection.return_value.document.assert_called_with(
            "event_20260306T143000Z"
        )
        call_data = (
            mock_db.collection.return_value.document.return_value.set.call_args[0][0]
        )
        assert call_data["type"] == "event"
        assert call_data["old_status"] == "closed"
        assert call_data["new_status"] == "open"

    @patch("garage_monitor.firestore_store.FirestoreClient")
    def test_get_recent_events(self, mock_client_cls):
        mock_db = MagicMock()
        mock_client_cls.return_value = mock_db
        now = datetime.now(timezone.utc)

        doc1 = MagicMock()
        doc1.to_dict.return_value = {
            "type": "event",
            "timestamp": now,
            "old_status": "closed",
            "new_status": "open",
        }
        doc2 = MagicMock()
        doc2.to_dict.return_value = {
            "type": "event",
            "timestamp": now - timedelta(hours=1),
            "old_status": "open",
            "new_status": "closed",
        }
        mock_query = MagicMock()
        mock_query.stream.return_value = [doc1, doc2]
        mock_db.collection.return_value.where.return_value.order_by.return_value.limit.return_value = mock_query

        store = FirestoreStore("test-project")
        events = store.get_recent_events(limit=5)

        assert len(events) == 2
        assert events[0]["new_status"] == "open"
        assert events[1]["new_status"] == "closed"
