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
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class GarageState:
    current_status: GarageStatus = GarageStatus.UNKNOWN
    last_check_time: datetime | None = None
    last_change_time: datetime | None = None
    consecutive_errors: int = 0
    last_reminder_time: datetime | None = None
    last_image_hash: str | None = None
    muted_until: datetime | None = None
    last_final_warning_sent: bool = False
    pending_status: GarageStatus | None = None
    pending_count: int = 0


@dataclass
class UsageStats:
    period: str
    function_invocations: int = 0
    gemini_calls: int = 0
    gemini_input_tokens: int = 0
    gemini_output_tokens: int = 0
    firestore_reads: int = 0
    firestore_writes: int = 0
    garage_openings: int = 0
