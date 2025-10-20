# translation.py (Mit Global Model Load Lock & Config)

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import logging

# NEUE Imports
from config import settings
from state import models_cache, is_job_cancelled, MODEL_LOAD_LOCK

logger = logging.getLogger(__name__)

# NEU: Model Konfiguration aus config
MODEL_ID = settings.TRANSLATION_MODEL_ID
BATCH_SIZE = settings.DEFAULT_BATCH_SIZE  

LANG_MAP = {
    "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "it": "Italian",
}


class JobCancelledException(Exception):
    """Wird geworfen wenn ein Job abgebrochen wurde."""
    pass

# ... (Rest der Datei bleibt unverändert) ...
def check_cancellation(job_id: str):
    """Prüft ob Job gecancelt wurde und wirft Exception."""
    if is_job_cancelled(job_id):
        logger.info(f"[CANCEL] Translation für Job {job_id} abgebrochen")
        raise JobCancelledException(f"Job {job_id} wurde vom Benutzer abgebrochen")


def load_rosetta_model(device: str = "cuda:0"):
    """
    Lädt das Rosetta-Modell (Thread-sicher mit Global Lock).
    """
    model_cache_key = f"rosetta:{MODEL_ID}:{device}"
    
    cached_model = models_cache.get(model_cache_key)
    if cached_model:
        logger.info(f"[MODEL] Verwende gecachtes Rosetta Modell auf {device}.")
        return cached_model

    with MODEL_LOAD_LOCK:
        cached_model = models_cache.get(model_cache_key)
        if cached_model:
            logger.info(f"[MODEL] Verwende (inzwischen) gecachtes Rosetta Modell auf {device}.")
            return cached_model

        logger.info(f"[MODEL] Lade Rosetta Übersetzungsmodell ({MODEL_ID}) für {device}...")
        
        dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
        model, tokenizer = None, None

        try:
            logger.info(f"   Mode: {str(dtype).split('.')[-1].upper()} (PERFORMANCE OPTIMIZED)")
            logger.info(f"   Attention: {attn_implementation} (TURBO)")
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model.to(device)

        except (ImportError, RuntimeError) as e:
            logger.warning(f"[WARNUNG] Flash Attention 2 fehlgeschlagen: {e}. Fallback auf Standard.")
            if model is not None:
                try: model.cpu(); del model; torch.cuda.empty_cache()
                except: pass
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                model.to(device)
            except Exception as fallback_error:
                logger.error(f"[FEHLER] Fallback fehlgeschlagen: {fallback_error}", exc_info=True)
                return None, None
        
        except Exception as e:
            logger.error(f"[FEHLER] KRITISCHER FEHLER beim Laden: {e}", exc_info=True)
            if model is not None:
                try: model.cpu(); del model; torch.cuda.empty_cache()
                except: pass
            return None, None
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        models_cache.set(model_cache_key, (model, tokenizer))
        logger.info(f"[MODEL] Rosetta Modell geladen und gecacht auf {device}.")
            
        return model, tokenizer


def clean_translation_output(text: str) -> str:
    """Bereinigt Translation-Output."""
    return text.strip()


def translate_segments_with_rosetta(
    job_id: str,
    segments: List[Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    device: str = "cuda:0"
) -> List[Dict[str, Any]]:
    """
    Übersetzt Segmente mit Rosetta - mit Cancel-Checkpoints zwischen Batches.
    """
    if not segments:
        return segments

    check_cancellation(job_id)

    model, tokenizer = load_rosetta_model(device)
    if model is None or tokenizer is None:
        raise RuntimeError("Rosetta-Modell konnte nicht geladen werden.")

    target_language_name = LANG_MAP.get(target_lang, target_lang.capitalize())
    
    logger.info(f"[TRANSLATE] Starte Übersetzung nach {target_language_name} (Batch: {BATCH_SIZE}) auf {device}")
    
    texts_to_translate = [s.get("text", "").strip() for s in segments if s.get("text", "").strip()]
    valid_indices = [i for i, s in enumerate(segments) if s.get("text", "").strip()]
    total_segments = len(texts_to_translate)
    logger.info(f"   Zu übersetzen: {total_segments} Segmente")
    
    check_cancellation(job_id)
    
    all_translations = []
    total_batches = (total_segments + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()
    
    for i in range(0, total_segments, BATCH_SIZE):
        check_cancellation(job_id)
        
        batch_start = time.time()
        batch_texts = texts_to_translate[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        logger.info(f"   Batch {batch_num}/{total_batches} ({len(batch_texts)} Segmente)...")

        messages_batch = []
        for text in batch_texts:
            system_content = (
                f"Translate the user's text to {target_language_name}.\n"
                "Provide the final translation immediately without any other text."
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": text}
            ]
            messages_batch.append(messages)

        prompts = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
            for msg in messages_batch
        ]

        tokenizer.padding_side = 'left'
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, 
            truncation=True, max_length=1024
        ).to(model.device)

        input_token_length = inputs.input_ids.shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024, 
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False
            )

        generated_token_ids = outputs[:, input_token_length:]
        decoded_translations = tokenizer.batch_decode(
            generated_token_ids, skip_special_tokens=True
        )
        
        for translation in decoded_translations:
            all_translations.append(clean_translation_output(translation))
        
        batch_time = time.time() - batch_start
        logger.info(f"   Batch OK ({batch_time:.1f}s)")

    total_time = time.time() - start_time
    avg_per_segment = total_time / total_segments if total_segments > 0 else 0
    logger.info(f"[TRANSLATE] Übersetzung abgeschlossen in {total_time:.1f}s ({avg_per_segment:.2f}s pro Segment)")

    check_cancellation(job_id)

    final_segments = segments.copy()
    translation_idx = 0
    for i, segment in enumerate(final_segments):
        if i in valid_indices:
            if translation_idx < len(all_translations):
                final_segments[i]["translated_text"] = all_translations[translation_idx]
                translation_idx += 1
        else:
            final_segments[i]["translated_text"] = ""
            
    return final_segments