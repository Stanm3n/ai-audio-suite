# models.py (Mit Enums & erweitertem Job-Modell)

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

# --- NEU: API-Enums für Validierung ---

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    COMPLETE_WITH_ERRORS = "complete_with_errors"
    CANCELLED = "cancelled"
    ERROR = "error"

class ExportFormat(str, Enum):
    srt = "srt"
    txt = "txt"

class OutputType(str, Enum):
    original = "original"
    translated = "translated"
    both = "both"

class ValidModelEnum(str, Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large_v2 = "large-v2"
    large_v3 = "large-v3"

class ValidTranslationLangEnum(str, Enum):
    de = "de"
    en = "en"
    es = "es"
    fr = "fr"
    it = "it"

# --------------------------------------

class JobResult(BaseModel):
    """Ergebnis eines abgeschlossenen Jobs."""
    segments: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None
    total_processing_duration: Optional[float] = None
    timing: Optional[Dict[str, float]] = None
    translation: Optional[Dict[str, float]] = None
    translation_error: Optional[str] = None
    error: Optional[str] = None  # Für Fehler oder Cancel-Messages


class Job(BaseModel):
    """Vollständige Job-Repräsentation (intern)."""
    job_id: str
    status: JobStatus
    created_at: float
    completed_at: Optional[float] = None
    started_at: Optional[float] = None # Hinzugefügt für bessere Nachverfolgung
    filename: str
    model: str
    language: str
    device: str
    translate_to: Optional[str] = None
    result: Optional[JobResult] = None
    access_token: str
    
    # --- NEU: Wichtig für den Worker ---
    temp_audio_path: Optional[str] = None


class JobResponse(BaseModel):
    """Was der User nach Job-Erstellung zurück bekommt."""
    job_id: str
    status: JobStatus
    access_token: str


class JobStatusResponse(BaseModel):
    """Was der User bei /results/{job_id} bekommt."""
    job_id: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    filename: str
    model: str
    language: str
    device: str
    translate_to: Optional[str] = None
    result: Optional[JobResult] = None
    # Kein access_token hier (aus Sicherheitsgründen)