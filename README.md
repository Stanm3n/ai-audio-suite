# AI Audio Suite: Performante Transkriptions-API

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.119-blue?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-blue?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Dieses Projekt ist eine ausfallsichere High-Throughput-API zur Transkription und Übersetzung sehr großer Audiodateien (5GB+). Es handelt sich um eine Architektur-Demo, die entwickelt wurde, um die Kernherausforderungen beim Betrieb schwerer KI-Modelle in einer Produktionsumgebung zu lösen: **Stabilität, Effizienz und Datenschutz.**

Das System läuft zu 100 % lokal und verarbeitet sensible Audiodaten (DSGVO-konform), ohne dass diese jemals Ihr Netzwerk verlassen.

![AI Audio Suite in Aktion](./assets/GIFDemo.gif)

---

## Das Kernproblem (Anwendungsfall "Kanzlei")

Ein typischer Anwender (z. B. eine Anwaltskanzlei) muss täglich Dutzende sensibler Diktate verarbeiten. Die Anforderungen sind klar:
1.  **Datenschutz:** Die Audiodateien dürfen das lokale Netzwerk nicht verlassen (DSGVO).
2.  **Stabilität:** Das System darf *nicht* abstürzen, nur weil 10 Mitarbeiter gleichzeitig Dateien hochladen.
3.  **Effizienz:** Die teure GPU muss voll ausgelastet werden, ohne dass der VRAM überläuft.

Beim Ausführen von KI-Modellen wie Whisper sind die häufigsten Fehlerquellen VRAM-Abstürze (OOM-Fehler) und ineffiziente Ressourcennutzung. Diese API löst beides durch eine robuste **Producer-Consumer-Architektur.**

| Problem-Stellung (Fehlerquelle) | Architektonische Lösung (Implementierung) |
| :--- | :--- |
| **VRAM OOM-Abstürze** durch parallele Anfragen. | **Dedizierte Job-Warteschlangen pro GPU.** (`state.py`) Für jede verfügbare GPU (`cuda:0`, `cuda:1`, ...) wird eine eigene, Thread-sichere `queue.Queue` erstellt. Jobs werden *seriell* pro Gerät abgearbeitet, was garantiert, dass nur eine schwere Aufgabe pro GPU läuft. |
| **Ineffizientes Laden von Modellen** für jeden Job. | **VRAM-optimierter LRU-Cache.** (`state.py:VRAMModelCache`) Ein globaler, Thread-sicherer Cache hält die zuletzt genutzten Modelle im VRAM. Wenn der Cache voll ist, wird das am längsten nicht genutzte Modell intelligent auf die CPU verschoben (`.to('cpu')`) und der Cache geleert, um OOM-Fehler zu verhindern. |
| **Konflikte bei Modell-Ladevorgängen** (Race Conditions). | **Globales Lade-Lock für Modelle.** (`state.py:MODEL_LOAD_LOCK`) Ein einziges globales Lock verhindert, dass *sogar unterschiedliche* GPUs gleichzeitig neue Modelle laden. Dies vermeidet VRAM-Spitzen während des Ladevorgangs, die zu Abstürzen führen könnten. |
| **"Zombie-Jobs"** oder Nutzerfehler. | **Kontrollierter Job-Abbruch.** (`state.py:request_job_cancellation`) Ein API-Endpunkt (`/cancel`) markiert einen Job. Der Worker prüft dieses Flag an mehreren Prüfpunkten (z.B. vor der Übersetzung) und bricht den Job sauber ab. |

### Performance-Benchmark (RTX 3090)

Diese Architektur beweist ihre Effizienz:
**~3,5 Stunden Audiomaterial (mit mehreren Sprechern) werden in ~8 Minuten transkribiert und ausgerichtet (aligned).**

---

## Projektstruktur & Architektur

Die Codebasis ist nach Verantwortlichkeiten getrennt und spiegelt das Producer-Consumer-Muster wider.

```
AI_AUDIO_SUITE_GITHUB/
│
├── api/
│   └── jobs.py                    # (Producer) FastAPI-Endpunkte, Uploads & Job-Queuing
│
├── core/
│   ├── exporters.py               # (Consumer) Erzeugt SRT/TXT-Dateien aus Ergebnissen
│   └── transcription.py           # (Consumer) Die Haupt-Worker-Logik (WhisperX-Pipeline)
│
├── static/                        # Frontend (HTML/CSS/JS)
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── assets/                        # Demo-Dateien
│   ├── 3min25demoEN.wav
│   ├── 3min25demoEN_OutputType.both.srt
│   └── GIFDemo.gif
│
├── config.py                      # Zentrale Konfiguration (Dateigrößen, Modellnamen)
├── main.py                        # App-Entrypoint, startet Worker & FastAPI
├── models.py                      # Pydantic-Modelle & Enums für API-Validierung
├── requirements.txt
├── state.py                       # ⭐ KERNSTÜCK: Globaler Zustand (Caches, Queues, Locks)
└── translation.py                 # (Consumer) Logik für das Rosetta-Übersetzungsmodell
```

### Zentrale Architekturmerkmale

1.  **Producer (FastAPI):** Die API-Endpunkte in `api/jobs.py` sind extrem schlank. Sie validieren **nur** die Eingabe, speichern die Datei temporär und fügen eine `job_id` zur korrekten Geräte-Warteschlange hinzu. Sie führen *keine* KI-Berechnungen durch.
2.  **Consumer (Worker):** Beim Start (`main.py`) wird für *jede* verfügbare GPU ein dedizierter `threading.Thread` gestartet. Diese Worker-Schleife (`core/transcription.py:run_worker_loop`) holt `job_id`s aus ihrer Warteschlange und führt die Transkriptions-Pipeline nacheinander aus.
3.  **Zentralisierter, Thread-sicherer Zustand:** Der gesamte Anwendungszustand (Job-Status `jobs`, Caches `models_cache`, Queues `job_queues`, Locks `MODEL_LOAD_LOCK`) wird global in `state.py` verwaltet. `threading.RLock` und `threading.Lock` stellen die Datenintegrität über alle Worker-Threads hinweg sicher.
4.  **Intelligentes Model-Caching (`VRAMModelCache`):** Die `_cleanup_model`-Methode des Caches verschiebt verschiedene Modelltypen (WhisperX, Rosetta, Align) korrekt zurück auf die CPU und ruft `torch.cuda.empty_cache()` auf – ein Beweis für korrektes VRAM-Management in PyTorch.

---

## Tech-Stack & Demonstrierte Konzepte

* **Backend:** FastAPI, Pydantic
* **KI & ML:** PyTorch, `whisperx`, Transformers
* **Concurrency (Nebenläufigkeit):** `threading`, `queue`, `RLock`, `Lock`
* **Architektur:** Producer-Consumer-Muster, LRU-Cache
* **Kernkonzepte:** VRAM-Management, Vermeidung von Race Conditions, Zustandsmanagement (State Management), asynchrone Job-Verarbeitung.

---

## Lokale Installation & Start

#### 1. Systemanforderungen
* NVIDIA GPU mit **CUDA Toolkit 12.1**
* **FFmpeg** (muss installiert und im System-PATH verfügbar sein)
* **Python 3.10+**
* (Windows) Microsoft C++ Build Tools

#### 2. Python-Setup & Start

```bash
# 1. Repository klonen
git clone https://github.com/Stanm3n/ai-audio-suite.git
cd ai-audio-suite

# 2. Virtuelle Umgebung erstellen & aktivieren
# Windows
python -m venv venv
venv\Scripts\activate
# macOS / Linux
# python3 -m venv venv
# source venv/bin/activate

# 3. WICHTIG: PyTorch FÜR CUDA 12.1 ZUERST INSTALLIEREN
# (Dies muss VOR der requirements.txt geschehen)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Alle anderen Abhängigkeiten installieren
pip install -r requirements.txt

# 5. Server starten (Entwicklermodus mit Auto-Reload)
uvicorn main:app --reload
#
# Oder für Produktion:
# python main.py
```

Der Server ist jetzt unter `http://127.0.0.1:8000` erreichbar.

---

## API-Nutzung (cURL-Beispiele)

Sie können die API direkt über die integrierte Dokumentation unter `http://127.0.0.1:8000/docs` testen oder die Endpunkte direkt ansteuern.

#### 1. Systemstatus prüfen

Sehen Sie, welche GPUs (`available_devices`) erkannt wurden.

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/system/status"
```

#### 2. Job absenden (Transkribieren & Übersetzen)

```bash
# Ersetzen Sie 'cuda:0' mit Ihrem Gerät aus dem /status-Endpunkt
curl -X POST "http://127.0.0.1:8000/api/v1/transcribe" \
     -F "file=@/pfad/zu/meiner/audio.mp3" \
     -F "model_name=large-v3" \
     -F "language=auto" \
     -F "device=cuda:0" \
     -F "translate_to=de"
```

**Antwort (JSON):**

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "queued",
  "access_token": "e5f6g7h8-..."
}
```

**Notieren Sie sich `job_id` und `access_token`!**

#### 3. Job-Status abfragen

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/results/a1b2c3d4-..."
```

#### 4. Ergebnis herunterladen (SRT-Format)

Benötigt den `access_token` zur "Authentifizierung".

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/results/a1b2c3d4-.../download?format=srt&output=translated&access_token=e5f6g7h8-..." \
     -o "translated_subs.srt"
```

#### 5. Job abbrechen

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/jobs/a1b2c3d4-.../cancel?access_token=e5f6g7h8-..."
```

---

## 🚀 Mögliche Erweiterungen

Dieses Projekt ist als Architektur-Proof-of-Concept konzipiert. Potenzielle Erweiterungen:

- **Diarization (Sprechererkennung):** Integration von `pyannote.audio` zur automatischen Sprecher-Trennung
- **Batch-Processing:** Ordner-Upload mit automatischer Verarbeitung aller Dateien
- **Webhooks:** Benachrichtigung bei Job-Completion (E-Mail, Slack, Discord)
- **Docker-Container:** One-Click-Deployment mit `docker-compose`
- **Alternative Modelle:** Austauschbare Translation-Models (NLLB, Opus-MT)
- **Web-UI Verbesserungen:** Drag & Drop, Progress-Bars, Live-Logs

---

## 📄 Lizenz

Dieser Code steht unter der **MIT License** (siehe [LICENSE](LICENSE)).

### Verwendete Modelle & deren Lizenzen:
- **WhisperX** (Whisper-Modelle): [Apache 2.0 License](https://github.com/openai/whisper/blob/main/LICENSE)
- **Rosetta** (basierend auf Gemma 3): [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

Die Lizenzen der verwendeten KI-Modelle sind unabhängig von der MIT-Lizenz dieses Codes zu beachten.

---

## 🤝 Kontakt

Bei Fragen zum Projekt oder Interesse an einer Zusammenarbeit können Sie mich gerne auf [LinkedIn](https://www.linkedin.com/in/stanislav-eirich/) kontaktieren.