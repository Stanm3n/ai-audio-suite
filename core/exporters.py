# core/exporters.py


from typing import List, Dict, Any, Literal

def format_timestamp(seconds: float) -> str:
    """Konvertiert Sekunden in das SRT-Zeitformat (00:00:00,000)"""
    assert seconds >= 0, "Negative Zeitstempel sind nicht erlaubt"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    seconds = milliseconds // 1000
    milliseconds %= 1000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def get_text_for_segment(
    segment: Dict[str, Any], 
    output_type: Literal["original", "translated", "both"]
) -> str:
    """Holt den gewünschten Text basierend auf der Auswahl."""
    original_text = segment.get("text", "").strip()
    translated_text = segment.get("translated_text", "").strip()

    if output_type == "translated" and translated_text:
        return translated_text
    elif output_type == "both" and translated_text:
        # Stelle sicher, dass beide Teile existieren
        if original_text and translated_text:
             return f"{original_text}\n{translated_text}"
        elif translated_text:
            return translated_text
        else:
            return original_text # Fallback
    else:
        # Fallback auf Original, wenn "translated" gewählt, aber nicht vorhanden
        return original_text

def generate_srt(
    segments: List[Dict[str, Any]], 
    output_type: Literal["original", "translated", "both"]
) -> str:
    """Generiert den Inhalt einer SRT-Datei aus den Segmenten."""
    srt_content = []
    for i, segment in enumerate(segments, start=1):
        start_time = segment.get("start")
        end_time = segment.get("end")
        
        if start_time is None or end_time is None:
            continue

        text = get_text_for_segment(segment, output_type)
        if not text:
            continue
            
        srt_content.append(str(i))
        srt_content.append(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}")
        srt_content.append(text)
        srt_content.append("")  # Leerzeile als Trenner

    return "\n".join(srt_content)

def generate_txt(
    segments: List[Dict[str, Any]], 
    output_type: Literal["original", "translated", "both"]
) -> str:
    """Generiert den Inhalt einer TXT-Datei aus den Segmenten."""
    txt_content = []
    for segment in segments:
        text = get_text_for_segment(segment, output_type)
        if text:
            # Füge Zeilenumbrüche für "both" ein, um die Lesbarkeit zu verbessern
            txt_content.append(text.replace("\n", "\n\n"))

    return "\n\n".join(txt_content)