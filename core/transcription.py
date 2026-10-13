# core/transcription.py (KORRIGIERT FÜR BUG 1)

import whisperx
import torch
import time
import os
import logging
import queue
from typing import Dict, Any, Optional, Tuple

# NEUE Imports
from config import settings
from models import Job, JobStatus, JobResult
from state import (
    jobs, jobs_lock, models_cache,
    is_job_cancelled, clear_cancellation,
    MODEL_LOAD_LOCK
)
from translation import translate_segments_with_rosetta, JobCancelledException as TranslationJobCancelledException


BATCH_SIZE = settings.DEFAULT_BATCH_SIZE
MAX_JOBS_IN_MEMORY = settings.MAX_JOBS_IN_MEMORY
logger = logging.getLogger(__name__)


class JobCancelledException(Exception):
    """Wird geworfen wenn ein Job abgebrochen wurde."""
    pass

# ... (check_cancellation, cleanup_old_jobs, run_worker_loop, get_whisper_model, get_align_model bleiben unverändert) ...
def check_cancellation(job_id: str):
    if is_job_cancelled(job_id):
        logger.info(f"[CANCEL] Job {job_id} wird abgebrochen (Checkpoint erreicht)")
        raise JobCancelledException(f"Job {job_id} wurde vom Benutzer abgebrochen")


def cleanup_old_jobs() -> None:
    try:
        if len(jobs) > MAX_JOBS_IN_MEMORY: 
            job_list_snapshot = list(jobs.values())
            completed_jobs = [(j.job_id, j.completed_at or 0) 
                             for j in job_list_snapshot
                             if j.status not in [JobStatus.PROCESSING, JobStatus.QUEUED]]
            
            if len(completed_jobs) > MAX_JOBS_IN_MEMORY * 0.75:
                completed_jobs.sort(key=lambda x: x[1])
                to_remove_count = len(jobs) - (MAX_JOBS_IN_MEMORY // 2)
                if to_remove_count > 0:
                    for jid, _ in completed_jobs[:to_remove_count]:
                        jobs.pop(jid, None)
                    logger.info(f"[CLEANUP] {to_remove_count} alte Jobs aus dem Speicher entfernt.")
    except Exception as e:
        logger.error(f"Fehler beim Aufräumen der Jobs: {e}", exc_info=True)


def run_worker_loop(device: str, q: queue.Queue):
    logger.info(f"[{device}] Worker-Loop startet... Wartet auf Jobs.")
    
    while True:
        job_id = None
        job_data: Optional[Job] = None
        temp_audio_path = None
        
        try:
            job_id = q.get() 
            
            with jobs_lock:
                job_data = jobs.get(job_id)
            
            if not job_data:
                logger.warning(f"[{device}] Hat Job {job_id} gezogen, aber er existiert nicht mehr.")
                continue
            
            temp_audio_path = job_data.temp_audio_path
            
            if is_job_cancelled(job_id) or job_data.status == JobStatus.CANCELLED:
                logger.info(f"[{device}] Job {job_id} wurde in der Queue abgebrochen. Überspringe.")
                clear_cancellation(job_id)
                if temp_audio_path and os.path.exists(temp_audio_path):
                    try:
                        os.remove(temp_audio_path)
                        logger.info(f"[CLEANUP] Temporäre Datei für gecancelten Queue-Job gelöscht: {temp_audio_path}")
                    except Exception as e:
                         logger.warning(f"[CLEANUP] Konnte Temp-Datei nicht löschen: {e}")
                continue

            job_data.status = JobStatus.PROCESSING
            job_data.started_at = time.time()
            
            logger.info(f"[{device}] Startet Verarbeitung von Job {job_id}...")

            _process_job(
                job_id=job_id,
                job_data=job_data
            )

        except Exception as e:
            logger.error(f"[{device}] Kritischer Fehler im Worker-Loop bei Job {job_id}: {e}", exc_info=True)
            if job_id and job_data:
                job_data.status = JobStatus.ERROR
                job_data.result = JobResult(error="Interner Worker-Fehler.")
                job_data.completed_at = time.time()
        finally:
            if job_id:
                q.task_done()


def get_whisper_model(model_name: str, device: str) -> Any:
    model_cache_key = f"whisper:{model_name}:{device}"
    model = models_cache.get(model_cache_key)
    
    if model:
        logger.info(f"[{device}] Verwende gecachtes Whisper-Modell: {model_name}")
        return model

    with MODEL_LOAD_LOCK:
        model = models_cache.get(model_cache_key)
        if model:
            logger.info(f"[{device}] Verwende (inzwischen) gecachtes Whisper-Modell: {model_name}")
            return model
        
        logger.info(f"[{device}] Lade Whisper-Modell: {model_name}...")
        device_type, *di_list = device.split(":")
        device_index = int(di_list[0]) if di_list and device_type != "cpu" else 0
        
        model = whisperx.load_model(
            model_name, 
            device=device_type, 
            device_index=device_index,
            compute_type="float16" if "cuda" in device else "int8",
        )
        models_cache.set(model_cache_key, model)
        logger.info(f"[{device}] Whisper-Modell {model_name} geladen und gecacht.")
        return model


def get_align_model(language_code: str, device: str) -> Tuple[Any, Dict]:
    align_model_key = f"align:{language_code}:{device}"
    align_model_tuple = models_cache.get(align_model_key)
    
    if align_model_tuple:
        logger.info(f"[{device}] Verwende gecachtes Alignment-Modell für '{language_code}'.")
        return align_model_tuple

    with MODEL_LOAD_LOCK:
        align_model_tuple = models_cache.get(align_model_key)
        if align_model_tuple:
            logger.info(f"[{device}] Verwende (inzwischen) gecachtes Alignment-Modell für '{language_code}'.")
            return align_model_tuple
            
        logger.info(f"[{device}] Lade Alignment-Modell für '{language_code}'...")
        align_model_tuple = whisperx.load_align_model(
            language_code=language_code, device=device
        )
        models_cache.set(align_model_key, align_model_tuple)
        logger.info(f"[{device}] Alignment-Modell für '{language_code}' geladen und gecacht.")
        return align_model_tuple


# --- HIER IST FIX #1 (run_transcription_pipeline) ---
def run_transcription_pipeline(
    job_id: str,
    audio_path: str,
    model_name: str, 
    language: Optional[str], 
    device: str,
) -> Dict[str, Any]:
    """
    Führt die WhisperX-Pipeline aus, lädt Modelle dynamisch.
    """
    check_cancellation(job_id)
    whisper_model = get_whisper_model(model_name, device)
    
    check_cancellation(job_id)
    audio = whisperx.load_audio(audio_path)
    language_to_use = language if language and language != "auto" else None
    
    check_cancellation(job_id)
    
    logger.info(f"[{device}] Starte Transkription mit {model_name} (Sprache: {language_to_use or 'auto'})...")
    transcribe_start = time.time()
    result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE, language=language_to_use)
    transcribe_time = time.time() - transcribe_start
    logger.info(f"[{device}] Transkription abgeschlossen in {transcribe_time:.2f}s")

    check_cancellation(job_id)

    if result.get("segments"):
        # HIER IST DIE ÄNDERUNG:
        # Finde die *tatsächlich* verwendete Sprache (Erkennung ODER Fallback)
        detected_language = result.get('language') or language_to_use
        
        if not detected_language:
             logger.warning(f"[{device}] Konnte Sprache nicht erkennen, überspringe Alignment.")
             result['timing'] = {'transcribe_duration': transcribe_time, 'align_duration': 0}
             result['language'] = None # Explizit 'None' setzen
             return result

        # Speichere die endgültige Sprache im Ergebnis, damit die Übersetzung sie nutzen kann
        result['language'] = detected_language
        # ENDE DER ÄNDERUNG

        # Alignment-Modell dynamisch laden
        align_model, metadata = get_align_model(detected_language, device)
        
        check_cancellation(job_id)
        
        align_start = time.time()
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device, 
            return_char_alignments=False
        )
        align_time = time.time() - align_start
        logger.info(f"[{device}] Alignment abgeschlossen in {align_time:.2f}s")
        
        # 'language' wurde durch whisperx.align() wieder überschrieben,
        # stelle sicher, dass es unsere endgültige Sprache bleibt.
        result['language'] = detected_language
        result['timing'] = {
            'transcribe_duration': transcribe_time, 
            'align_duration': align_time
        }
        
    return result


# --- HIER IST FIX #2 (_process_job) ---
def _process_job(
    job_id: str, 
    job_data: Job
) -> None:
    """Die Haupt-Worker-Funktion mit Cancel-Handling."""
    
    audio_path = job_data.temp_audio_path
    model_name = job_data.model
    language = job_data.language # Das ist die *Benutzer*-Eingabe ("auto" oder "de")
    device = job_data.device
    translate_to = job_data.translate_to
    
    job_status = JobStatus.COMPLETE
    
    try:
        start_time = time.time()
        
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {audio_path}")

        # Pipeline mit Job-ID für Cancel-Checks
        result = run_transcription_pipeline(
            job_id, audio_path, model_name, language, device
        )
        
        check_cancellation(job_id)
        
        if translate_to and result.get('segments'):
            
            # HIER IST DIE ÄNDERUNG:
            # Wir nehmen jetzt *immer* die Sprache aus dem Pipeline-Ergebnis.
            # Diese ist garantiert die Sprache der Segmente.
            source_language = result.get('language')
            # (Alte, fehlerhafte Zeile: 
            #  source_language = language if language and language != "auto" else result.get('language'))
            
            if source_language and source_language != translate_to:
                logger.info(f"[{device}] Starte Übersetzung: {source_language} -> {translate_to}")
                trans_start = time.time()
                try:
                    translated_segments = translate_segments_with_rosetta(
                        job_id,
                        result['segments'], source_language, translate_to, device
                    )
                    trans_time = time.time() - trans_start
                    result['segments'] = translated_segments
                    result['translation'] = {'duration': trans_time}
                    logger.info(f"[{device}] Übersetzung abgeschlossen in {trans_time:.2f}s")
                
                except (JobCancelledException, TranslationJobCancelledException):
                    raise 
                
                except Exception as trans_error:
                    logger.warning(f"[{device}] Übersetzung fehlgeschlagen: {trans_error}", exc_info=True)
                    result['translation_error'] = f"Übersetzung fehlgeschlagen: {type(trans_error).__name__}"
                    job_status = JobStatus.COMPLETE_WITH_ERRORS
            
            elif not source_language:
                logger.warning(f"[{device}] Übersetzung übersprungen: Quellsprache konnte nicht ermittelt werden (Ergebnis war None).")
            
            else: # source_language == translate_to
                logger.info(f"[{device}] Übersetzung übersprungen: Quell- und Zielsprache sind identisch ({source_language}).")
            # ENDE DER ÄNDERUNG

        total_duration = time.time() - start_time
        result["total_processing_duration"] = total_duration
        
        # ... (Rest der Funktion: Job-Objekt aktualisieren, Cleanup) ...
        job_data.status = job_status
        job_data.result = result 
        job_data.completed_at = time.time()
            
        with jobs_lock:
            cleanup_old_jobs() 
            
        logger.info(f"[{device}] Job {job_id} abgeschlossen (Status: {job_status.value}) in {total_duration:.2f}s")
        
    except (JobCancelledException, TranslationJobCancelledException) as e:
        logger.info(f"[{device}] Job {job_id} wurde abgebrochen: {e}")
        job_data.status = JobStatus.CANCELLED
        job_data.result = JobResult(error="Job wurde vom Benutzer abgebrochen")
        job_data.completed_at = time.time()
        
        with jobs_lock:
            cleanup_old_jobs()
    
    except Exception as e:
        logger.error(f"[{device}] Job {job_id} fehlgeschlagen: {e}", exc_info=True)
        user_error_message = f"Verarbeitung fehlgeschlagen: {type(e).__name__}. Details in den Server-Logs."
        job_status = JobStatus.ERROR
        
        job_data.status = job_status
        job_data.result = JobResult(error=user_error_message)
        job_data.completed_at = time.time()
            
        with jobs_lock:
            cleanup_old_jobs() 
            
    finally:
        clear_cancellation(job_id)
        
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"[CLEANUP] Temporäre Audio-Datei gelöscht: {audio_path}")
            except Exception as e:
                logger.warning(f"[CLEANUP] Konnte Audio-Datei nicht löschen: {e}")
