from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class GarageStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


@dataclass
class GeminiAnalysisResult:
    status: GarageStatus
    confidence: float
    reasoning: str


@dataclass
class GarageState:
    current_status: GarageStatus = GarageStatus.UNKNOWN
    last_check_time: datetime | None = None
    last_change_time: datetime | None = None
    consecutive_errors: int = 0
