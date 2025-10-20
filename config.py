# config.py

import os
from pydantic_settings import BaseSettings
from typing import List, Set

class Settings(BaseSettings):
    """
    Zentrale Konfiguration für die AI Audio Suite.
    Werte werden von Umgebungsvariablen überschrieben (z.B. MAX_FILE_SIZE_BYTES).
    """
    
    # --- System & API Limits ---
    MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024 * 1024  # 5 GB
    UPLOAD_CHUNK_SIZE: int = 8 * 1024 * 1024         # 8 MB
    MAX_JOBS_IN_MEMORY: int = 100
    MAX_CONCURRENT_JOBS: int = 100 # (Wird aktuell nicht hart erzwungen, aber gut zu haben)
    TEMP_FILE_PREFIX: str = "ai_audio_suite_"
    
    # --- AI Model Konfiguration ---
    VRAM_CACHE_MAX_SIZE: int = 8
    DEFAULT_BATCH_SIZE: int = 16
    
    # --- Whisper Models ---
    VALID_MODELS: List[str] = [
        "tiny", "base", "small", "medium", 
        "large-v2", "large-v3"
    ]
    
    # --- Translation Models & Languages ---
    TRANSLATION_MODEL_ID: str = "yanolja/YanoljaNEXT-Rosetta-4B-2510"
    VALID_TRANSLATION_LANG: List[str] = ["de", "en", "es", "fr", "it"]
    
    @property
    def VALID_LANGUAGES(self) -> Set[str]:
        """Kombinierte Menge aller gültigen Sprach-Inputs."""
        # Set für schnelle Lookups
        return set(self.VALID_TRANSLATION_LANG) | {"auto", None}

# Globale Instanz, die in anderen Modulen importiert wird
settings = Settings()