# state.py 

import torch
import threading
import queue
from typing import Dict, Any, Set, List
from collections import OrderedDict
import logging

# NEUE Imports
from config import settings
from models import Job, JobStatus  

logger = logging.getLogger(__name__)

# --- Job-Management ---
jobs: Dict[str, Job] = {} 
jobs_lock = threading.RLock()

# --- Job-Abbruch-Mechanismus ---
cancelled_jobs: Set[str] = set()
cancel_lock = threading.Lock()

def request_job_cancellation(job_id: str) -> bool:
    """Markiert einen Job zum Abbruch."""
    with cancel_lock:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return False
            # Zugriff über Objekt-Property 
            if job.status not in [JobStatus.PROCESSING, JobStatus.QUEUED]:
                return False
            cancelled_jobs.add(job_id)
            logger.info(f"[CANCEL] Job {job_id} zum Abbruch markiert")
            return True

def is_job_cancelled(job_id: str) -> bool:
    """Prüft ob ein Job abgebrochen werden soll."""
    with cancel_lock:
        return job_id in cancelled_jobs

def clear_cancellation(job_id: str):
    """Entfernt Cancel-Flag nach Job-Ende."""
    with cancel_lock:
        cancelled_jobs.discard(job_id)

# --- Globaler Lock für das Laden von Modellen ---
MODEL_LOAD_LOCK = threading.Lock()


# --- Dedizierte Job-Queues ---
job_queues: Dict[str, queue.Queue] = {}
AVAILABLE_DEVICES: List[str] = []

def _initialize_queues():
    """
    Füllt 'job_queues' und 'AVAILABLE_DEVICES' basierend auf der Hardware.
    'cpu' wird nur als Fallback hinzugefügt, wenn keine GPU verfügbar ist.
    """
    global AVAILABLE_DEVICES
    
    device_list = []
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"[STATE] {device_count} CUDA-Gerät(e) gefunden.")
        for i in range(device_count):
            device_list.append(f"cuda:{i}")
    else:
        logger.warning("[STATE] Kein CUDA-Gerät gefunden. Fallback auf CPU.")
        device_list.append("cpu")
    
    for device in device_list:
        job_queues[device] = queue.Queue()
        
    AVAILABLE_DEVICES = device_list
    logger.info(f"[STATE] Verfügbare Geräte-Queues initialisiert: {AVAILABLE_DEVICES}")

_initialize_queues()


# --- VRAM-BEWUSSTER MODELL-CACHE ---
class VRAMModelCache:
    def __init__(self, max_size=8):
        # max_size wird von config übergeben
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.RLock() 
        logger.info(f"[CACHE] VRAM Model Cache initialisiert (max_size={self.max_size})")

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key, value):
        """Fügt ein Modell hinzu und räumt atomar auf."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return 

            self.cache[key] = value
            
            while len(self.cache) > self.max_size:
                old_key, old_value = self.cache.popitem(last=False)
                logger.info(f"[CACHE] VRAM Cache: Entferne altes Modell '{old_key}'")
                
                try:
                    self._cleanup_model(old_key, old_value)
                except Exception as e:
                    logger.warning(f"Fehler beim Bereinigen von Modell {old_key}: {e}")
    
    def _cleanup_model(self, key, value):
        """
        Bereinigt ein Modell richtig aus dem VRAM.
        Unterscheidet zwischen verschiedenen Modell-Typen.
        """
        try:
            # FALL 1: WhisperX ASR Model
            if hasattr(value, 'model') and hasattr(value.model, 'to'):
                logger.info(f"[CACHE] Verschiebe WhisperX Modell '{key}' auf CPU...")
                value.model.to('cpu')
                del value.model
                del value
            
            # FALL 2: Alignment Model (Tuple: model, metadata)
            elif isinstance(value, tuple) and len(value) == 2:
                model, metadata = value
                if hasattr(model, 'to'):
                    logger.info(f"[CACHE] Verschiebe Alignment Modell '{key}' auf CPU...")
                    model.to('cpu')
                del value
            
            # FALL 3: Rosetta Translation Model (Tuple: model, tokenizer)
            elif isinstance(value, tuple) and len(value) == 2:
                model, tokenizer = value
                if hasattr(model, 'cpu'):
                    logger.info(f"[CACHE] Verschiebe Rosetta Modell '{key}' auf CPU...")
                    model.cpu()
                del model
                del tokenizer
                del value
            
            # FALL 4: Unbekanntes Modell
            else:
                logger.warning(f"[CACHE] Unbekannter Modell-Typ für '{key}', einfaches Löschen...")
                del value
            
            # VRAM freigeben
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"[CACHE] VRAM Cache geleert für '{key}'")
                
        except Exception as e:
            logger.error(f"[CACHE] Fehler beim Cleanup von '{key}': {e}", exc_info=True)
            try:
                del value
            except:
                pass

    def clear(self):
        """Leert den kompletten Cache beim Shutdown."""
        with self.lock:
            keys = list(self.cache.keys())
            for key in keys:
                try:
                    val = self.cache[key]
                    self._cleanup_model(key, val)
                    self.cache.pop(key, None)
                except Exception as e:
                    logger.warning(f"Fehler beim Löschen von {key}: {e}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("[CACHE] VRAM Cache komplett geleert.")


# Globale Instanz (nutzt config)
models_cache = VRAMModelCache(max_size=settings.VRAM_CACHE_MAX_SIZE)