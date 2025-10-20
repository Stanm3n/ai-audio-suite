// script.js 

document.addEventListener('DOMContentLoaded', () => {
    // --- GLOBALE VARIABLEN ---
    let selectedFile = null;
    const activePolls = new Map();
    let mediaRecorder = null;
    let audioChunks = [];
    const jobDataStore = new Map(); 

    // --- DOM-ELEMENTE SAMMELN ---
    const elements = {
        // DATEI-VERARBEITUNG
        dropZone: document.getElementById('drop-zone'),
        fileInput: document.getElementById('file-input'),
        fileNameDisplay: document.getElementById('file-name-display'),
        submitBtn: document.getElementById('submit-btn'),
        uploadForm: document.getElementById('upload-form'),
        
        // AUFNAHME-STEUERUNG
        recordBtn: document.getElementById('record-btn'),
        stopRecordBtn: document.getElementById('stop-record-btn'),
        recordStatus: document.getElementById('record-status'),

        // JOBS & ERGEBNISSE
        jobsContainer: document.getElementById('jobs-container'),
        placeholder: document.querySelector('#jobs-container .placeholder'),
        jobCardTemplate: document.getElementById('job-card-template'),
    };

    // --- INITIALISIERUNG ---
    populateDropdowns();
    setupEventListeners();

    // =========================================================================
    // EVENT LISTENERS
    // =========================================================================
    function setupEventListeners() {
        // Event Listeners für Datei-Verarbeitung
        elements.dropZone.addEventListener('click', () => elements.fileInput.click());
        elements.fileInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));
        elements.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); elements.dropZone.classList.add('drag-over'); });
        elements.dropZone.addEventListener('dragleave', () => elements.dropZone.classList.remove('drag-over'));
        elements.dropZone.addEventListener('drop', (e) => { e.preventDefault(); elements.dropZone.classList.remove('drag-over'); handleFileSelect(e.dataTransfer.files[0]); });
        elements.uploadForm.addEventListener('submit', handleFormSubmit);

        // Event Listeners für Aufnahme-Buttons
        if (elements.recordBtn) elements.recordBtn.addEventListener('click', startRecording);
        if (elements.stopRecordBtn) elements.stopRecordBtn.addEventListener('click', stopRecording);
    }

    // =========================================================================
    // DATEI-VERARBEITUNG & AUFNAHME
    // =========================================================================

    function handleFileSelect(file) {
        if (!file) return;
        selectedFile = file;
        elements.fileNameDisplay.textContent = `Ausgewählt: ${file.name}`;
        elements.submitBtn.disabled = false;
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            audioChunks = [];
            mediaRecorder.addEventListener('dataavailable', event => audioChunks.push(event.data));
            mediaRecorder.start();
            elements.recordBtn.disabled = true;
            elements.stopRecordBtn.disabled = false;
            elements.recordStatus.textContent = 'Nimmt auf...';
            elements.recordStatus.style.color = 'var(--error-color)';
        } catch (err) {
            console.error("Fehler beim Mikrofonzugriff:", err);
            elements.recordStatus.textContent = 'Mikrofon-Zugriff verweigert.';
            elements.recordStatus.style.color = 'var(--error-color)';
        }
    }

    function stopRecording() {
        if (!mediaRecorder) return;
        mediaRecorder.addEventListener('stop', () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const recordingFile = new File([audioBlob], `aufnahme-${timestamp}.webm`, { type: 'audio/webm' });
            handleFileSelect(recordingFile);
            if (elements.submitBtn) elements.submitBtn.click();
            elements.recordBtn.disabled = false;
            elements.stopRecordBtn.disabled = true;
            elements.recordStatus.textContent = 'Aufnahme beendet, Job startet...';
            elements.recordStatus.style.color = 'var(--success-color)';
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            mediaRecorder = null;
        });
        mediaRecorder.stop();
    }
    
    async function handleFormSubmit(e) {
        e.preventDefault();
        if (!selectedFile) return alert("Bitte wählen Sie zuerst eine Datei aus.");

        elements.submitBtn.disabled = true;
        elements.submitBtn.textContent = 'Job wird gestartet...';

        const formData = new FormData(elements.uploadForm);
        formData.set('file', selectedFile);
        
        const deviceSelect = document.getElementById('device-select');
        formData.set('device', deviceSelect.value);
        
        if (formData.get('translate_to') === "") {
            formData.delete('translate_to');
        }
        
        formData.delete('gpu_id');

        try {
            const response = await fetch('/api/v1/transcribe', { method: 'POST', body: formData });
            
            if (response.status !== 202) {
                let message = "Unbekannter Server-Fehler.";
                try {
                    const errorData = await response.json();
                    
                    if (errorData.detail) {
                        if (Array.isArray(errorData.detail)) {
                            // FastAPI 422 Validierungsfehler (z.B. falsches Enum)
                            message = errorData.detail.map(e => {
                                const field = e.loc && e.loc.length > 1 ? `Feld '${e.loc[1]}'` : 'Eingabe';
                                return `${field}: ${e.msg}`;
                            }).join('; ');
                        } else {
                            // Normaler HTTPException (z.B. 413 "Datei zu groß")
                            message = errorData.detail;
                        }
                    } else {
                        message = JSON.stringify(errorData); // Fallback
                    }
                } catch (e) {
                    message = `Server-Antwort: ${response.statusText} (Status: ${response.status})`;
                }
                throw new Error(message);
            }
            
            const data = await response.json(); // data.status ist jetzt 'queued'
            
            jobDataStore.set(data.job_id, { 
                filename: selectedFile.name, 
                access_token: data.access_token 
            }); 
            
            createNewJobCard(data.job_id, selectedFile.name);
            startPolling(data.job_id);

            selectedFile = null;
            elements.fileNameDisplay.textContent = '';
            elements.fileInput.value = ''; 

        } catch (error) {
            console.error("Fehler beim Job-Start:", error);
            alert(`Job konnte nicht gestartet werden:\n${error.message}`);
        } finally {
            elements.submitBtn.disabled = false;
            elements.submitBtn.textContent = 'Transkription starten';
        }
    }
    
    // =========================================================================
    // JOB-CANCELLATION
    // =========================================================================
    
    async function cancelJob(jobId) {
        const storedData = jobDataStore.get(jobId);
        if (!storedData || !storedData.access_token) {
            alert("Fehler: Access Token nicht gefunden.");
            return;
        }

        const card = document.querySelector(`.job-card[data-job-id="${jobId}"]`);
        const cancelBtn = card?.querySelector('.cancel-btn');
        
        if (cancelBtn) {
            cancelBtn.disabled = true;
            cancelBtn.textContent = 'Abbruch läuft...';
        }

        try {
            const response = await fetch(`/api/v1/jobs/${jobId}/cancel?access_token=${storedData.access_token}`, {
                method: 'POST'
            });

            if (!response.ok) {
                let errorDetail = `Abbruch fehlgeschlagen (Status ${response.status})`;
                try {
                    const errorData = await response.json();
                    errorDetail = errorData.detail || JSON.stringify(errorData);
                } catch (e) { /* Ignore */ }
                throw new Error(errorDetail);
            }

            const result = await response.json();
            console.log('[CANCEL]', result);

            if (result.cancelled) {
                // Status-Update wird vom Polling übernommen
                if (cancelBtn) {
                    cancelBtn.textContent = 'Wird abgebrochen...';
                }
            } else {
                alert(result.message || 'Job konnte nicht abgebrochen werden.');
                if (cancelBtn) {
                    cancelBtn.disabled = false;
                    cancelBtn.textContent = 'Abbrechen';
                }
            }

        } catch (error) {
            console.error("Cancel fehlgeschlagen:", error);
            alert(`Abbruch fehlgeschlagen:\n${error.message}`);
            if (cancelBtn) {
                cancelBtn.disabled = false;
                cancelBtn.textContent = 'Abbrechen';
            }
        }
    }
    
    // =========================================================================
    // INITIALISIERUNG & JOB-HANDLING
    // =========================================================================

    async function populateDropdowns() {
         try {
            const response = await fetch('/api/v1/system/status');
            const data = await response.json();
            
            const modelSelect = document.getElementById('model-select');
            modelSelect.innerHTML = '';
            data.supported_models.forEach(model => { 
                const option = new Option(model, model); 
                if (model === 'medium') option.selected = true; 
                modelSelect.appendChild(option); 
            });
            
            const translateSelect = document.getElementById('translate-select');
            translateSelect.innerHTML = '<option value="">Nicht übersetzen</option>';
            data.supported_translation_languages.forEach(lang => { 
                translateSelect.appendChild(new Option(`${lang.toUpperCase()}`, lang)); 
            });
            
            const deviceSelect = document.getElementById('device-select');
            deviceSelect.innerHTML = '';
            
            if (data.available_devices && data.available_devices.length > 0) {
                let bestDevice = null;
                data.available_devices.forEach(device => {
                    let deviceName = device.toUpperCase();
                    if (device.startsWith('cuda') && !bestDevice) {
                        bestDevice = device;
                        deviceName = `${deviceName} (Empfohlen)`;
                    }
                    if (device === 'cpu') deviceName = "CPU (Langsam)";
                    
                    const option = new Option(deviceName, device);
                    deviceSelect.appendChild(option);
                });
                if (bestDevice) {
                    deviceSelect.value = bestDevice;
                }
            } else {
                deviceSelect.appendChild(new Option('CPU', 'cpu'));
            }

        } catch (error) { 
            console.error("Fehler beim Laden der System-Infos:", error); 
        }
    }

    function createNewJobCard(jobId, filename) {
        if (elements.placeholder) elements.placeholder.hidden = true;
        const cardClone = elements.jobCardTemplate.content.cloneNode(true);
        const newCard = cardClone.querySelector('.job-card');
        newCard.dataset.jobId = jobId;
        newCard.querySelector('.job-filename').textContent = filename;
        
        const cancelBtn = newCard.querySelector('.cancel-btn');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => cancelJob(jobId));
        }
        
        elements.jobsContainer.prepend(newCard);
    }

    function startPolling(jobId) {
        if (activePolls.has(jobId)) return;
        const intervalId = setInterval(async () => {
            try {
                const response = await fetch(`/api/v1/results/${jobId}`);
                if (!response.ok) {
                    if (response.status === 404) stopPolling(jobId, intervalId); 
                    return;
                }
                const data = await response.json();
                
                const storedData = jobDataStore.get(jobId) || {};
                jobDataStore.set(jobId, { ...storedData, ...data }); 

                if (['complete', 'error', 'complete_with_errors', 'cancelled'].includes(data.status)) {
                    stopPolling(jobId, intervalId);
                }
                updateJobCard(data, jobId);
            } catch (error) {
                console.error(`Polling error for job ${jobId}:`, error);
            }
        }, 5000); // Polling-Intervall (5 Sekunden)
        activePolls.set(jobId, intervalId);
    }
    
    function stopPolling(jobId, intervalId) {
        clearInterval(intervalId);
        activePolls.delete(jobId);
    }
    
    function updateJobCard(data, jobId) {
        const card = document.querySelector(`.job-card[data-job-id="${jobId}"]`);
        if (!card) return;

        const statusDiv = card.querySelector('.job-status');
        const statusText = statusDiv.querySelector('span');
        const loader = statusDiv.querySelector('.loader');
        const cancelBtn = card.querySelector('.cancel-btn');
        
        if (loader) loader.hidden = true;
        
        statusDiv.className = 'job-status'; // Reset classes
        
        switch (data.status) {
            case 'queued':
                statusDiv.classList.add('queued');
                statusText.textContent = 'In Warteschlange...';
                if (loader) loader.hidden = false;
                if (cancelBtn) cancelBtn.hidden = false;
                break;
            case 'processing':
                statusDiv.classList.add('processing');
                statusText.textContent = 'Verarbeite...';
                if (loader) loader.hidden = false;
                if (cancelBtn) cancelBtn.hidden = false;
                break;
            case 'complete':
                statusDiv.classList.add('complete');
                statusText.textContent = 'Abgeschlossen';
                if (cancelBtn) cancelBtn.hidden = true;
                break;
            case 'complete_with_errors':
                statusDiv.classList.add('warning');
                statusText.textContent = 'Abgeschlossen (mit Fehlern)';
                if (cancelBtn) cancelBtn.hidden = true;
                break;
            case 'error':
                statusDiv.classList.add('error');
                statusText.textContent = 'Fehlgeschlagen';
                if (cancelBtn) cancelBtn.hidden = true;
                break;
            case 'cancelled':
                statusDiv.classList.add('cancelled');
                statusText.textContent = 'Abgebrochen';
                if (cancelBtn) cancelBtn.hidden = true;
                break;
            default:
                statusText.textContent = data.status;
        }

        if (['complete', 'error', 'complete_with_errors', 'cancelled'].includes(data.status)) {
            const totalTimeDiv = card.querySelector('.job-total-time');
            const totalTime = data.result?.total_processing_duration;
            if (totalTime) {
                totalTimeDiv.textContent = `Erledigt in ${totalTime.toFixed(2)}s`;
            } else if (data.status === 'cancelled' || data.status === 'error') {
                 totalTimeDiv.textContent = `Gestoppt`;
            }
            
            renderResult(card, data, jobId);
        }
    }
    
    function renderResult(card, data, jobId) {
        const body = card.querySelector('.job-card-body');
        body.hidden = false;
        const output = card.querySelector('.result-output');
        const viewOptionsContainer = card.querySelector('.view-options');
        viewOptionsContainer.innerHTML = ''; 

        if (data.status === 'cancelled') {
            output.textContent = `Job wurde abgebrochen: ${data.result?.error || 'Vom Benutzer abgebrochen'}`;
            card.querySelector('.download-buttons').hidden = true;
            card.querySelector('.job-timing-breakdown').hidden = true;
            return;
        }

        if (data.status === 'error') {
            output.textContent = `Fehler: ${data.result?.error || 'Unbekannter Fehler'}`;
            card.querySelector('.download-buttons').hidden = true; 
            card.querySelector('.job-timing-breakdown').hidden = true;
            return;
        }
        
        card.querySelector('.download-buttons').hidden = false; 

        // Timing-Aufschlüsselung
        const breakdownDiv = card.querySelector('.job-timing-breakdown');
        let breakdownHtml = '<ul>';
        const timings = data.result?.timing;
        if (timings?.transcribe_duration) breakdownHtml += `<li>Transkription: ${timings.transcribe_duration.toFixed(2)}s</li>`;
        if (timings?.align_duration) breakdownHtml += `<li>Alignment (Zeitstempel): ${timings.align_duration.toFixed(2)}s</li>`;
        
        const translation = data.result?.translation;
        if (translation?.duration) breakdownHtml += `<li>Übersetzung: ${translation.duration.toFixed(2)}s</li>`;
        
        breakdownHtml += '</ul>';
        breakdownDiv.innerHTML = breakdownHtml;

        const segments = data.result?.segments;
        if (!segments) {
            output.textContent = 'Keine Segmente gefunden.';
            card.querySelector('.download-buttons').hidden = true;
            return;
        }

        const hasTranslation = data.status === 'complete' && segments.some(s => s.translated_text && s.translated_text.trim() !== '');
        const translationFailed = data.status === 'complete_with_errors' && data.result?.translation_error;

        if (hasTranslation) {
            viewOptionsContainer.innerHTML = `
                <div class="radio-group">
                    <label class="radio-label"><input type="radio" name="view-mode-${jobId}" value="both" checked><span>Beides</span></label>
                    <label class="radio-label"><input type="radio" name="view-mode-${jobId}" value="translated"><span>Übersetzung</span></label>
                    <label class="radio-label"><input type="radio" name="view-mode-${jobId}" value="original"><span>Original</span></label>
                </div>`;
            viewOptionsContainer.querySelectorAll('input[type="radio"]').forEach(radio => {
                radio.addEventListener('change', () => output.textContent = generateSrtContent(segments, card));
            });
        } else if (translationFailed) {
            viewOptionsContainer.innerHTML = `<span class="error-text">Übersetzung fehlgeschlagen: ${data.result.translation_error}</span>`;
        }

        output.textContent = generateSrtContent(segments, card); 

        // Download-Buttons neu binden, um alte Event Listener zu entfernen
        const downloadSrtBtn = card.querySelector('.download-srt');
        const downloadTxtBtn = card.querySelector('.download-txt');
        
        const newSrtBtn = downloadSrtBtn.cloneNode(true);
   
        downloadSrtBtn.parentNode.replaceChild(newSrtBtn, downloadSrtBtn);
        
        newSrtBtn.addEventListener('click', () => triggerDownload(jobId, card, 'srt')); 

        const newTxtBtn = downloadTxtBtn.cloneNode(true);
        downloadTxtBtn.parentNode.replaceChild(newTxtBtn, downloadTxtBtn);
        newTxtBtn.addEventListener('click', () => triggerDownload(jobId, card, 'txt')); 
    }

    function generateSrtContent(segments, card) {
        const modeRadio = card.querySelector('input[name^="view-mode"]:checked');
        const mode = modeRadio ? modeRadio.value : 'original';
        return segments.map((seg, i) => {
            const start = formatSrtTime(seg.start);
            const end = formatSrtTime(seg.end);
            let text = '';
            const originalText = seg.text?.trim() ?? '';
            const translatedText = seg.translated_text?.trim() ?? '';

            if (mode === 'both' && translatedText) text = `${originalText}\n${translatedText}`;
            else if (mode === 'translated' && translatedText) text = translatedText;
            else text = originalText;
            
            return text ? `${i + 1}\n${start} --> ${end}\n${text}\n` : null; 
        }).filter(Boolean).join('\n'); 
    }
    
    function formatSrtTime(seconds) {
        if (typeof seconds !== 'number' || isNaN(seconds) || seconds < 0) return '00:00:00,000';
        const date = new Date(seconds * 1000);
        const hours = String(Math.floor(seconds / 3600)).padStart(2, '0');
        const minutes = String(date.getUTCMinutes()).padStart(2, '0');
        const secs = String(date.getUTCSeconds()).padStart(2, '0');
        const ms = String(date.getUTCMilliseconds()).padStart(3, '0');
        return `${hours}:${minutes}:${secs},${ms}`;
    }

    async function triggerDownload(jobId, card, format) {
        const storedData = jobDataStore.get(jobId);
        if (!storedData || !storedData.access_token) {
            alert("Fehler: Job-Daten oder Access Token nicht gefunden. Bitte Seite neu laden.");
            return;
        }

        const modeRadio = card.querySelector('input[name^="view-mode"]:checked');
        const outputMode = modeRadio ? modeRadio.value : 'original';
        const accessToken = storedData.access_token;

        const downloadUrl = `/api/v1/results/${jobId}/download?format=${format}&output=${outputMode}&access_token=${accessToken}`;

        try {
            const response = await fetch(downloadUrl);

            if (!response.ok) {
                let errorDetail = `Download fehlgeschlagen (Status ${response.status})`;
                try { 
                    const errorData = await response.json(); 
                    errorDetail = errorData.detail || JSON.stringify(errorData); 
                } catch(e) { /* Ignore */ }
                throw new Error(errorDetail);
            }

            const disposition = response.headers.get('Content-Disposition');
            let filename = `${jobId}_${outputMode}.${format}`; 
            if (disposition && disposition.indexOf('attachment') !== -1) {
                const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                const matches = filenameRegex.exec(disposition);
                if (matches != null && matches[1]) { 
                    filename = matches[1].replace(/['"]/g, '');
                }
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = filename; 
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();

        } catch (error) {
            console.error("Download fehlgeschlagen:", error);
            alert(`Download fehlgeschlagen:\n${error.message}`);
        }
    }
});