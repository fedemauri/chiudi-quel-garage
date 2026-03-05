from google.cloud.firestore_v1 import Client as FirestoreClient
from datetime import datetime, timezone
from garage_monitor.models import GarageState, GarageStatus


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
        return GarageState(
            current_status=GarageStatus(data["current_status"]),
            last_check_time=data.get("last_check_time"),
            last_change_time=data.get("last_change_time"),
            consecutive_errors=data.get("consecutive_errors", 0),
        )

    def save_state(self, state: GarageState) -> None:
        """Salva/aggiorna il documento 'state'."""
        self._doc("state").set({
            "current_status": state.current_status.value,
            "last_check_time": state.last_check_time,
            "last_change_time": state.last_change_time,
            "consecutive_errors": state.consecutive_errors,
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
