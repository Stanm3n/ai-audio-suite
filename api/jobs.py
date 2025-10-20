# api/jobs.py 

import uuid
import tempfile
import os
import torch
import time
import re
import logging
from fastapi import (
    APIRouter, UploadFile, File, Form, 
    HTTPException, Response, status
)
from typing import Optional, Dict, Any, List

# NEUE Imports
from config import settings
import state
from state import (
    jobs, jobs_lock, 
    request_job_cancellation,
    job_queues,
    AVAILABLE_DEVICES
)
from models import (
    Job, JobResponse, JobStatusResponse, JobStatus, JobResult, # JobResult ist wichtig
    ValidModelEnum, ValidTranslationLangEnum,
    ExportFormat, OutputType
)
from core.exporters import generate_srt, generate_txt 

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_JOBS_IN_MEMORY = settings.MAX_JOBS_IN_MEMORY 

# ... (sanitize_filename, submit_transcription_job, cancel_job, get_job_result, list_all_jobs, get_system_status bleiben) ...
def sanitize_filename(filename: str) -> str:
    """Bereinigt einen Dateinamen von gefährlichen Zeichen."""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)
    return filename[:100]


@router.post(
    "/api/v1/transcribe", 
    tags=["Jobs"], 
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def submit_transcription_job(
    file: UploadFile = File(...),
    model_name: ValidModelEnum = Form(ValidModelEnum.medium),
    language: Optional[str] = Form(None),
    device: str = Form(..., description=f"Gerät für die Verarbeitung. Verfügbar: {AVAILABLE_DEVICES}"),
    translate_to: Optional[ValidTranslationLangEnum] = Form(None)
) -> JobResponse:
    
    lang_to_use = language or "auto"
    if lang_to_use not in settings.VALID_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Ungültige Sprache. Wähle aus: {list(settings.VALID_LANGUAGES)}")

    if device not in AVAILABLE_DEVICES:
        raise HTTPException(
            status_code=400, 
            detail=f"Gerät '{device}' nicht verfügbar. Wähle aus: {AVAILABLE_DEVICES}"
        )

    with jobs_lock:
        if len(jobs) >= MAX_JOBS_IN_MEMORY:
            completed_jobs = [(jid, job.completed_at or 0) 
                             for jid, job in list(jobs.items()) 
                             if job.status not in [JobStatus.PROCESSING, JobStatus.QUEUED]]
            completed_jobs.sort(key=lambda x: x[1])
            to_remove = len(jobs) - (MAX_JOBS_IN_MEMORY // 2)
            if to_remove > 0:
                for jid, _ in completed_jobs[:to_remove]:
                    jobs.pop(jid, None)
            
            if len(jobs) >= MAX_JOBS_IN_MEMORY:
                raise HTTPException(status_code=503, detail="Server-Speicher für Jobs ist voll.")

    temp_file_path = None
    try:
        file_extension = os.path.splitext(file.filename)[1]
        temp_file_path = os.path.join(
            tempfile.gettempdir(), 
            f"{settings.TEMP_FILE_PREFIX}{uuid.uuid4()}{file_extension}"
        )
        
        total_size_bytes = 0
        with open(temp_file_path, 'wb') as f:
            while chunk := await file.read(settings.UPLOAD_CHUNK_SIZE):
                total_size_bytes += len(chunk)
                if total_size_bytes > settings.MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413, 
                        detail=f"Datei zu groß. Limit: {settings.MAX_FILE_SIZE_BYTES // 1024**3} GB"
                    )
                f.write(chunk)

        job_id = str(uuid.uuid4())
        access_token = str(uuid.uuid4())
        
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            created_at=time.time(),
            filename=file.filename,
            model=model_name.value, 
            language=lang_to_use,
            device=device,
            translate_to=translate_to.value if translate_to else None,
            access_token=access_token,
            temp_audio_path=temp_file_path 
        )
        
        with jobs_lock:
            jobs[job_id] = job

        state.job_queues[device].put(job_id)
        
        logger.info(f"[JOB] Job {job_id} in Queue für <{device}> aufgenommen. (Queue-Größe: {state.job_queues[device].qsize()})")
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            access_token=access_token
        )
        
    except HTTPException:
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except Exception as e: logger.warning(f"[CLEANUP] Konnte Temp-Datei nach Fehler nicht löschen: {e}")
        raise
        
    except Exception as e:
        logger.error(f"[FEHLER] Job-Erstellung fehlgeschlagen: {e}", exc_info=True)
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except Exception as e: logger.warning(f"[CLEANUP] Konnte Temp-Datei nach Fehler nicht löschen: {e}")
        raise HTTPException(status_code=500, detail="Fehler beim Verarbeiten des Uploads")


@router.post("/api/v1/jobs/{job_id}/cancel", tags=["Jobs"])
def cancel_job(job_id: str, access_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Bricht einen laufenden oder in der Queue wartenden Job ab.
    Erfordert den access_token zur Authentifizierung.
    """
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job nicht gefunden")
        
        if job.access_token != access_token:
            raise HTTPException(status_code=403, detail="Ungültiger oder fehlender Access Token")
        
        current_status = job.status
        if current_status not in [JobStatus.PROCESSING, JobStatus.QUEUED]:
            return {
                "job_id": job_id,
                "message": f"Job kann nicht abgebrochen werden (Status: {current_status})",
                "cancelled": False
            }
    
    success = request_job_cancellation(job_id)
    
    if success:
        logger.info(f"[CANCEL] Cancel-Request für Job {job_id} akzeptiert")
        
        if current_status == JobStatus.QUEUED:
            with jobs_lock:
                job.status = JobStatus.CANCELLED
                job.result = JobResult(error="Job wurde vom Benutzer abgebrochen (aus Queue entfernt)")
                job.completed_at = time.time()
            logger.info(f"[CANCEL] Job {job_id} direkt aus Queue als CANCELLED markiert.")

        return { "job_id": job_id, "message": "Job wird abgebrochen (oder wurde aus Queue entfernt)", "cancelled": True }
    else:
        return { "job_id": job_id, "message": "Job konnte nicht zum Abbruch markiert werden (Status evtl. geändert)", "cancelled": False }

@router.get("/api/v1/results/{job_id}", tags=["Jobs"], response_model=JobStatusResponse)
def get_job_result(job_id: str) -> JobStatusResponse:
    """Holt den Status und das Ergebnis eines einzelnen Jobs."""
    with jobs_lock: 
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job nicht gefunden")
        
        job_dict = job.model_dump(exclude={"access_token", "temp_audio_path"})
        
    return JobStatusResponse(**job_dict)


@router.get("/api/v1/jobs", tags=["Jobs"])
def list_all_jobs() -> List[Dict[str, Any]]:
    """Listet alle Jobs im Speicher auf."""
    with jobs_lock: 
        job_list = [
            j.model_dump(exclude={"access_token", "temp_audio_path"}) 
            for j in jobs.values()
        ]
    sorted_jobs = sorted(job_list, key=lambda j: j['created_at'], reverse=True)
    return sorted_jobs


@router.get("/api/v1/system/status", tags=["System"])
def get_system_status() -> Dict[str, Any]:
    """Gibt den Systemstatus und die Queue-Größen zurück."""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "id": i, "name": props.name,
                "memory_gb": round(props.total_memory / 1024**3, 2)
            })
            
    with jobs_lock: 
        active_jobs_count = len([j for j in jobs.values() if j.status == JobStatus.PROCESSING])
        queued_jobs_count = len([j for j in jobs.values() if j.status == JobStatus.QUEUED])
        total_jobs_count = len(jobs)
    
    queue_sizes = {device: q.qsize() for device, q in job_queues.items()}
    
    return {
        "cuda_available": torch.cuda.is_available(), 
        "gpus": gpus,
        "available_devices": AVAILABLE_DEVICES,
        "active_jobs_global": active_jobs_count,
        "queued_jobs_global": queued_jobs_count,
        "queue_sizes_per_device": queue_sizes,
        "total_jobs_in_memory": total_jobs_count, 
        "max_jobs_in_memory_limit": settings.MAX_JOBS_IN_MEMORY,
        "supported_models": settings.VALID_MODELS, 
        "supported_translation_languages": settings.VALID_TRANSLATION_LANG
    }


@router.get("/api/v1/results/{job_id}/download", tags=["Jobs"])
def download_job_result(
    job_id: str,
    format: ExportFormat = ExportFormat.srt,
    output: OutputType = OutputType.original,
    access_token: Optional[str] = None
) -> Response:
    """Stellt das Job-Ergebnis als Datei-Download bereit."""
    
   
    
    # Lokale Variablen initialisieren
    job_status: Optional[JobStatus] = None
    job_result: Optional[JobResult] = None # <-- Explizit als JobResult-Objekt typisiert
    job_filename: Optional[str] = None
    
    with jobs_lock: 
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job nicht gefunden")
        if job.access_token != access_token:
            raise HTTPException(status_code=403, detail="Ungültiger oder fehlender Access Token")
        
        job_status = job.status
        job_filename = job.filename
        
        
        # job.result ist ein dict (oder None). Wir müssen es explizit
        # in ein JobResult-Objekt parsen, um darauf zugreifen zu können.
        if job.result:
            try:
                # `model_validate` ist der Pydantic V2 Weg, um ein dict
                # sicher in ein Modell-Objekt umzuwandeln.
                job_result = JobResult.model_validate(job.result)
            except Exception as e:
                logger.error(f"Fehler beim Parsen von Job-Ergebnis {job_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Job-Ergebnis ist korrupt.")
    
    # ==================================================================
    # --- ENDE DER KORREKTUR ---
    # ==================================================================

    if job_status not in [JobStatus.COMPLETE, JobStatus.COMPLETE_WITH_ERRORS]:
        raise HTTPException(status_code=400, detail="Job ist nicht abgeschlossen.")

    if output in [OutputType.translated, OutputType.both]:
        # Ab hier ist job_result garantiert ein JobResult-Objekt
        if job_result and job_result.translation_error:
            raise HTTPException(
                status_code=400, 
                detail=f"Download fehlgeschlagen: Übersetzung ist fehlgeschlagen: {job_result.translation_error}"
            )

    # Ab hier ist der Zugriff .segments sicher
    segments = job_result.segments if job_result else None
    if not segments:
         raise HTTPException(status_code=400, detail="Job hat keine Segmente zum Exportieren.")

    original_filename = sanitize_filename(
        os.path.splitext(job_filename or "export")[0]
    )
    file_extension = "txt" if format == ExportFormat.txt else "srt"
    download_filename = f"{original_filename}_{output}.{file_extension}"

    content = ""
    if format == ExportFormat.txt:
        content = generate_txt(segments, output)
    else:
        content = generate_srt(segments, output)
        
    if not content:
        raise HTTPException(status_code=500, detail="Konnte leeren Inhalt nicht generieren.")

    return Response(
        content=content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{download_filename}"'
        }
    )