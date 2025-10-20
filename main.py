# main.py (Mit Worker-Thread-Start & Config)

import torch
import time
import os
import glob
import tempfile
import asyncio 
import logging 
import threading
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import List 

# NEUE Imports
from config import settings
from api import jobs as api_jobs
import state
from state import models_cache, jobs
# (TEMP_FILE_PREFIX wird unten direkt aus settings geholt)

from core.transcription import run_worker_loop 

# --- Logging Konfiguration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_audio_suite.log'), 
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)
# ... (Rest der Logging-Konfig unverändert) ...
logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.ERROR)
logging.getLogger("whisperx.vads.pyannote").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR) 
# -----------------------------------------------


# =============================================================================
# App Lifecycle & Cleanup
# =============================================================================

worker_threads: List[threading.Thread] = []

def cleanup_orphaned_files():
    """Löscht alle temporären Dateien von abgestürzten Jobs."""
    logger.info("[CLEANUP] Suche nach verwaisten temporären Dateien...")
    temp_dir = tempfile.gettempdir()
    # NEU: aus config
    pattern = os.path.join(temp_dir, f"{settings.TEMP_FILE_PREFIX}*") 
    count = 0
    
    for f in glob.glob(pattern):
        try:
            os.remove(f)
            count += 1
        except Exception as e:
            logger.warning(f"[CLEANUP] Konnte verwaiste Datei {f} nicht löschen: {e}")
            
    if count > 0:
        logger.info(f"[CLEANUP] {count} verwaiste Dateien entfernt.")
    else:
        logger.info("[CLEANUP] Keine verwaisten Dateien gefunden.")

async def periodic_cleanup_task():
    """Führt die Bereinigung stündlich aus."""
    while True:
        await asyncio.sleep(3600)  # 1 Stunde warten
        logger.info("[CLEANUP] Starte periodische Bereinigung verwaister Dateien...")
        try:
            cleanup_orphaned_files()
        except Exception as e:
            logger.error(f"Periodische Bereinigung fehlgeschlagen: {e}", exc_info=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup und Shutdown Events"""
    global worker_threads
    print("="*50)
    print("      AI AUDIO SUITE - BACKEND-ENGINE ") 
    print("="*50)
    
    # --- STARTUP ---
    logger.info("[STARTUP] Starte dedizierte Worker-Threads pro Gerät...")
    
    for device in state.AVAILABLE_DEVICES:
        q = state.job_queues[device]
        thread = threading.Thread(
            target=run_worker_loop, 
            args=(device, q), 
            daemon=True 
        )
        thread.start()
        worker_threads.append(thread)
        logger.info(f"[STARTUP] Worker-Thread für <{device}> gestartet.")
    
    logger.info(f"[STARTUP] {len(worker_threads)} Worker-Threads laufen.")
    
    cleanup_orphaned_files() 
    cleanup_task = asyncio.create_task(periodic_cleanup_task()) 
    print("="*50)
    
    yield # Hier läuft die App
    
    # --- SHUTDOWN ---
    print("\nServer fährt herunter, räume Caches auf...")
    cleanup_task.cancel() 
    models_cache.clear() 
    jobs.clear()
    
    logger.info("Worker-Threads werden beendet.")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Auf Wiedersehen!")


# =============================================================================
# FastAPI App-Initialisierung
# =============================================================================
app = FastAPI(
    title="AI Audio Suite API",
    version="3.1.0", # Versions-Bump!
    lifespan=lifespan 
)

app.include_router(api_jobs.router)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)