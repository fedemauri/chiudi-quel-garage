from google.cloud.firestore_v1 import Client as FirestoreClient
from google.cloud.firestore_v1.transforms import Increment
from datetime import datetime, timezone
from garage_monitor.models import GarageState, GarageStatus, UsageStats


class FirestoreStore:
    def __init__(self, project_id: str, collection: str = "garage_monitor"):
        self._db = FirestoreClient(project=project_id)
        self._collection = collection

    def _doc(self, doc_id: str):
        return self._db.collection(self._collection).document(doc_id)

    def get_state(self) -> GarageState | None:
        """Legge il documento 'state'. Ritorna None se non esiste."""
        snap = self._doc("state").get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        pending_raw = data.get("pending_status")
        pending_status = GarageStatus(pending_raw) if pending_raw else None
        return GarageState(
            current_status=GarageStatus(data["current_status"]),
            last_check_time=data.get("last_check_time"),
            last_change_time=data.get("last_change_time"),
            consecutive_errors=data.get("consecutive_errors", 0),
            last_reminder_time=data.get("last_reminder_time"),
            last_image_hash=data.get("last_image_hash"),
            muted_until=data.get("muted_until"),
            last_final_warning_sent=data.get("last_final_warning_sent", False),
            pending_status=pending_status,
            pending_count=data.get("pending_count", 0),
        )

    def save_state(self, state: GarageState) -> None:
        """Salva/aggiorna il documento 'state'."""
        self._doc("state").set({
            "current_status": state.current_status.value,
            "last_check_time": state.last_check_time,
            "last_change_time": state.last_change_time,
            "consecutive_errors": state.consecutive_errors,
            "last_reminder_time": state.last_reminder_time,
            "last_image_hash": state.last_image_hash,
            "muted_until": state.muted_until,
            "last_final_warning_sent": state.last_final_warning_sent,
            "pending_status": state.pending_status.value if state.pending_status else None,
            "pending_count": state.pending_count,
        })

    def get_blink_credentials(self) -> dict | None:
        """Legge credenziali Blink salvate. Ritorna None se non esistono."""
        snap = self._doc("blink_credentials").get()
        if not snap.exists:
            return None
        return snap.to_dict()

    def save_blink_credentials(self, credentials: dict) -> None:
        """Salva credenziali Blink con timestamp."""
        credentials["updated_at"] = datetime.now(timezone.utc)
        self._doc("blink_credentials").set(credentials)

    def get_usage_stats(self, period: str) -> UsageStats:
        """Legge statistiche di utilizzo per il periodo (es. '2026_03')."""
        snap = self._doc(f"usage_stats_{period}").get()
        if not snap.exists:
            return UsageStats(period=period)
        data = snap.to_dict()
        return UsageStats(
            period=period,
            function_invocations=data.get("function_invocations", 0),
            gemini_calls=data.get("gemini_calls", 0),
            gemini_input_tokens=data.get("gemini_input_tokens", 0),
            gemini_output_tokens=data.get("gemini_output_tokens", 0),
            firestore_reads=data.get("firestore_reads", 0),
            firestore_writes=data.get("firestore_writes", 0),
            garage_openings=data.get("garage_openings", 0),
        )

    def increment_usage(self, period: str, **counters: int) -> None:
        """Incrementa atomicamente i contatori di utilizzo."""
        doc_ref = self._doc(f"usage_stats_{period}")
        updates = {"period": period}
        updates.update({k: Increment(v) for k, v in counters.items() if v})
        doc_ref.set(updates, merge=True)

    def save_event(self, event: dict) -> None:
        """Save a status change event."""
        doc_id = f"event_{event['timestamp'].strftime('%Y%m%dT%H%M%SZ')}"
        event["type"] = "event"
        self._doc(doc_id).set(event)

    def get_recent_events(self, limit: int = 10) -> list[dict]:
        """Get most recent status change events."""
        from google.cloud.firestore_v1.base_query import FieldFilter

        query = (
            self._db.collection(self._collection)
            .where(filter=FieldFilter("type", "==", "event"))
            .order_by("timestamp", direction="DESCENDING")
            .limit(limit)
        )
        return [doc.to_dict() for doc in query.stream()]
