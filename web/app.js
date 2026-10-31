const fileInput = document.getElementById('audio-file');
const dropZone = document.getElementById('drop-zone');
const uploadTitle = document.getElementById('upload-title');
const fileDescription = document.getElementById('file-description');
const recordButton = document.getElementById('record-button');
const transcribeButton = document.getElementById('transcribe-button');
const copyButton = document.getElementById('copy-button');
const statusNode = document.getElementById('status');
const errorNode = document.getElementById('error');
const transcriptNode = document.getElementById('transcript');
const resultMeta = document.getElementById('result-meta');
const modelLabel = document.getElementById('model-label');
let selectedAudio = null;
let mediaRecorder = null;
let recordingTimer = null;

function setStatus(message) { statusNode.textContent = message; }
function showError(message) { errorNode.textContent = message; errorNode.hidden = false; }
function clearError() { errorNode.hidden = true; errorNode.textContent = ''; }

function wavFromFloat(samples) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const word = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
  word(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true);
  word(8, 'WAVE'); word(12, 'fmt '); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  word(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, value < 0 ? value * 32768 : value * 32767, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

async function prepareAudio(blob, name) {
  clearError();
  selectedAudio = null;
  transcribeButton.disabled = true;
  setStatus('Preparing audio…');
  if (blob.size > 8 * 1024 * 1024) throw new Error('Choose an audio file under 8 MB.');
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error('This browser cannot decode audio. Try a recent browser.');
  const context = new AudioContextClass();
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer());
    if (decoded.duration <= 0 || decoded.duration > 30) throw new Error('Audio must be between 0 and 30 seconds.');
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    const samples = rendered.getChannelData(0);
    const energy = samples.reduce((sum, value) => sum + value * value, 0) / samples.length;
    if (energy < 1e-10) throw new Error('The recording is silent or too quiet. Try again.');
    selectedAudio = wavFromFloat(samples);
    uploadTitle.textContent = name;
    fileDescription.textContent = `${decoded.duration.toFixed(1)} seconds · ready to transcribe`;
    setStatus('Audio ready. Transcribe when you are ready.');
    transcribeButton.disabled = false;
  } finally { await context.close(); }
}

async function selectAudio(blob, name) {
  try { await prepareAudio(blob, name); }
  catch (error) { setStatus('Audio could not be prepared.'); showError(error.message || 'Audio could not be decoded.'); }
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) selectAudio(file, file.name);
});
['dragenter', 'dragover'].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault(); dropZone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault(); dropZone.classList.remove('dragover');
}));
dropZone.addEventListener('drop', event => {
  const file = event.dataTransfer.files?.[0];
  if (file) selectAudio(file, file.name);
});

recordButton.addEventListener('click', async () => {
  clearError();
  if (mediaRecorder?.state === 'recording') { mediaRecorder.stop(); return; }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    showError('Microphone recording requires a supported browser over HTTPS. You can upload audio instead.'); return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.addEventListener('dataavailable', event => { if (event.data.size) chunks.push(event.data); });
    mediaRecorder.addEventListener('stop', async () => {
      clearTimeout(recordingTimer);
      stream.getTracks().forEach(track => track.stop());
      recordButton.textContent = 'Record with mic';
      recordButton.setAttribute('aria-pressed', 'false');
      await selectAudio(new Blob(chunks, { type: mediaRecorder.mimeType }), 'Microphone recording');
      mediaRecorder = null;
    }, { once: true });
    mediaRecorder.start();
    recordButton.textContent = 'Stop recording';
    recordButton.setAttribute('aria-pressed', 'true');
    setStatus('Recording… Speak clearly, then stop. Recording ends after 29 seconds.');
    recordingTimer = setTimeout(() => { if (mediaRecorder?.state === 'recording') mediaRecorder.stop(); }, 29000);
  } catch (error) { showError('Microphone access was unavailable. Check browser permission or upload a file.'); }
});

transcribeButton.addEventListener('click', async () => {
  if (!selectedAudio) return;
  clearError();
  transcribeButton.disabled = true;
  transcribeButton.textContent = 'Transcribing…';
  setStatus('The model is processing your recording.');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 90000);
  try {
    const response = await fetch('/api/v1/transcribe', {
      method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: selectedAudio, signal: controller.signal
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Transcription failed.');
    transcriptNode.textContent = data.text || 'No speech was recognized.';
    transcriptNode.classList.toggle('empty', !data.text);
    resultMeta.textContent = `${data.duration_seconds.toFixed(1)} s audio · ${data.processing_seconds.toFixed(1)} s processing · research output`;
    copyButton.disabled = !data.text;
    setStatus('Transcription complete. Review it carefully for errors.');
  } catch (error) {
    showError(error.name === 'AbortError' ? 'The request timed out. Try a shorter clip.' : error.message);
    setStatus('Transcription was not completed.');
  } finally {
    clearTimeout(timeout);
    transcribeButton.textContent = 'Transcribe audio →';
    transcribeButton.disabled = !selectedAudio;
  }
});

copyButton.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(transcriptNode.textContent); copyButton.textContent = 'Copied'; }
  catch { showError('Could not copy automatically. Select the transcript text instead.'); }
  setTimeout(() => { copyButton.textContent = 'Copy text'; }, 1800);
});

fetch('/api/v1/model').then(response => response.json()).then(data => {
  if (!data.model) throw new Error('Model unavailable');
  const label = data.model === 'openai/whisper-tiny.en' ? 'Whisper tiny English · pretrained baseline' : 'Whisper tiny English · research checkpoint';
  modelLabel.textContent = label;
}).catch(() => { modelLabel.textContent = 'Model is starting. Reload shortly.'; });
