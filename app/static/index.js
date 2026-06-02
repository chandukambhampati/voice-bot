// ============================================================================
// Frontend Controller — Advanced Cancer Center AI Telecalling Voicebot
// Production v3: Web Speech API primary (continuous), English default,
// dynamic Indian language support, mic permission handling
// ============================================================================

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let ws              = null;
let wsReconnectTimer = null;
let audioPlayer     = null;
let speechRecognizer = null;
let isCallActive    = false;
let isSpeaking      = false;
let isProcessing    = false;
// Streaming State
let audioQueue        = [];
let isAudioPlaying    = false;
let streamEnded       = false;
let isInterrupted     = false;
let currentAgentText  = null;
let shouldDisconnect  = false;
let holdMusicInterval = null;
let holdAudioCtx      = null;

// Ambient Noise
let ambientAudioCtx  = null;
let ambientNoiseNode = null;
let ambientGain      = null;

// Whisper fallback (mic_stream mode)
let mediaRecorder   = null;
let audioChunks     = [];
let userMediaStream = null;
let audioContext    = null;
let analyser        = null;
let vadInterval     = null;

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
const callToggleBtn       = document.getElementById("call-toggle-btn");
const callStatusText      = document.getElementById("call-status-text");
const languageSelect      = document.getElementById("language-select");
const inputModeSelect     = document.getElementById("input-mode-select");
const visualizerContainer = document.querySelector(".visualizer-container");
const avatar              = document.querySelector(".caller-avatar");
const transcriptContainer = document.getElementById("transcript-container");
const keyboardInput       = document.getElementById("keyboard-input");
const sendInputBtn        = document.getElementById("send-input-btn");
const ragContainer        = document.getElementById("rag-container");
const toolsLog            = document.getElementById("tools-log");

// CRM badge elements
const crmTags = {
    callerName:  document.getElementById("tag-caller-name"),
    relation:    document.getElementById("tag-relation"),
    cancer:      document.getElementById("tag-cancer"),
    symptoms:    document.getElementById("tag-symptoms"),
    authority:   document.getElementById("tag-authority"),
    travel:      document.getElementById("tag-travel"),
    temperature: document.getElementById("tag-temperature"),
    persona:     document.getElementById("tag-persona"),
    booking:     document.getElementById("tag-booking")
};

// ---------------------------------------------------------------------------
// Language → BCP-47 code (for Web Speech API lang attribute)
// Chrome supports en-IN, hi-IN, te-IN natively.
// ---------------------------------------------------------------------------
const LANG_CODES = {
    en_neerja:  "en-IN",
    en_prabhat: "en-IN",
    hi_swara:   "hi-IN",
    hi_madhur:  "hi-IN",
    te_shruti:  "te-IN",
    te_mohan:   "te-IN",
};

function getLangCode(voiceKey) {
    return LANG_CODES[voiceKey] || "en-IN";
}

// When language changes, update the live recognition if it's active
languageSelect.addEventListener("change", () => {
    const vk   = languageSelect.value;
    const code = getLangCode(vk);
    addToolLog("Language", "action", `Voice changed → ${code}. Restarting recognition.`);
    if (isCallActive && !isSpeaking) {
        stopListening();
        setTimeout(() => { if (isCallActive && !isSpeaking) startListening(); }, 200);
    }
});

// ---------------------------------------------------------------------------
// Mic Permission Handling
// Run on page load so the user sees their mic status immediately.
// ---------------------------------------------------------------------------
const micBanner   = document.getElementById("mic-status-banner");
const micLabel    = document.getElementById("mic-status-label");
const micIconEl   = document.getElementById("mic-icon");

function setMicStatus(state, text) {
    micBanner.className = "mic-status-banner mic-" + state;
    micLabel.textContent = text;
}

async function checkMicPermission() {
    setMicStatus("unknown", "Requesting mic access...");
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            },
            video: false
        });
        // Permission granted — stop the test stream immediately
        stream.getTracks().forEach(t => t.stop());
        setMicStatus("granted", "✓ Microphone is ready. Click 'Start Call' to begin.");
    } catch (err) {
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            setMicStatus("denied",
                "Mic BLOCKED. Click the 🔒 lock icon in Chrome's address bar → Site settings → Microphone → Allow → Refresh.");
        } else if (err.name === "NotFoundError") {
            setMicStatus("denied", "No microphone found. Please plug in a mic and try again.");
        } else {
            setMicStatus("denied", `Mic error: ${err.message}. Try refreshing the page.`);
        }
    }
}

// Check mic permission using the Permissions API first (non-blocking)
async function initMicStatus() {
    if (!navigator.permissions) {
        setMicStatus("unknown", "Click 'Test Mic' to check microphone access.");
        return;
    }
    try {
        const result = await navigator.permissions.query({ name: "microphone" });
        if (result.state === "granted") {
            setMicStatus("granted", "✓ Microphone is ready. Click 'Start Call' to begin.");
        } else if (result.state === "denied") {
            setMicStatus("denied",
                "Mic BLOCKED. Click the 🔒 lock icon → Site settings → Microphone → Allow → Refresh page.");
        } else {
            // "prompt" state — mic access will be asked when call starts
            setMicStatus("unknown", "Mic permission not yet granted. Click 'Start Call' — browser will ask for mic access.");
        }
        // Listen for permission changes (e.g., user grants from address bar)
        result.onchange = initMicStatus;
    } catch(e) {
        setMicStatus("unknown", "Click 'Test Mic' to verify microphone access.");
    }
}

// Run on page load
initMicStatus();

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------
callToggleBtn.addEventListener("click", toggleCall);
sendInputBtn.addEventListener("click", sendKeyboardInput);
keyboardInput.addEventListener("keydown", (e) => { if (e.key === "Enter") sendKeyboardInput(); });

// ---------------------------------------------------------------------------
// Call Controls
// ---------------------------------------------------------------------------
function toggleCall() {
    isCallActive ? endCall() : startCall();
}

function startCall() {
    isCallActive  = true;
    isSpeaking    = false;
    isProcessing  = false;
    callToggleBtn.classList.add("active");
    callStatusText.textContent = "Connecting...";
    callStatusText.classList.add("active");
    avatar.classList.add("pulsing");
    transcriptContainer.innerHTML = "";
    
    // Initialize AudioContext during user gesture to avoid browser autoplay blocks
    if (!pcmContext) {
        pcmContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    }
    if (pcmContext.state === 'suspended') pcmContext.resume();
    
    addToolLog("API", "action", "Initiating WebSocket connection to AI server...");
    connectWebSocket();
}

function endCall() {
    isCallActive = false;
    callToggleBtn.classList.remove("active");
    callStatusText.textContent = "Disconnected";
    callStatusText.classList.remove("active");
    avatar.classList.remove("pulsing");
    visualizerContainer.classList.remove("playing");
    keyboardInput.setAttribute("disabled", "true");
    sendInputBtn.setAttribute("disabled", "true");
    keyboardInput.value = "";
    if (audioPlayer) { audioPlayer.pause(); audioPlayer = null; }
    stopListening();
    stopHoldMusic();
    stopAmbientNoise();
    if (wsReconnectTimer) { clearTimeout(wsReconnectTimer); wsReconnectTimer = null; }
    if (ws) { try { ws.close(); } catch(e) {} ws = null; }
    addToolLog("Call Engine", "action", "Call ended.");
}

// ---------------------------------------------------------------------------
// WebSocket with auto-reconnect
// ---------------------------------------------------------------------------
function connectWebSocket() {
    if (wsReconnectTimer) { clearTimeout(wsReconnectTimer); wsReconnectTimer = null; }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/call`);

    ws.onopen = () => {
        callStatusText.textContent = "Connected";
        addToolLog("WebSocket", "success", "Connected to AI server.");
        keyboardInput.removeAttribute("disabled");
        sendInputBtn.removeAttribute("disabled");
        
        startAmbientNoise();

        // Send initial config — bot generates dynamic greeting based on name + voice
        const leadName = document.getElementById("lead-name-input").value.trim();
        const voiceKey = languageSelect.value;
        ws.send(JSON.stringify({ event: "start", lead_name: leadName, voice_key: voiceKey }));
    };

    let pcmDataBuffer = new Uint8Array(0);

    ws.onmessage = async (event) => {
        if (typeof event.data === "string") {
            try { handleServerMessage(JSON.parse(event.data)); }
            catch(e) { console.error("Bad message from server:", e); }
        } else if (event.data instanceof Blob) {
            // Binary audio chunk (PCM 16-bit)
            const arrayBuffer = await event.data.arrayBuffer();
            
            // Append to buffer
            let tmp = new Uint8Array(pcmDataBuffer.length + arrayBuffer.byteLength);
            tmp.set(pcmDataBuffer, 0);
            tmp.set(new Uint8Array(arrayBuffer), pcmDataBuffer.length);
            pcmDataBuffer = tmp;

            // Process even bytes
            let evenBytes = pcmDataBuffer.length - (pcmDataBuffer.length % 2);
            if (evenBytes > 0) {
                let chunkToProcess = pcmDataBuffer.buffer.slice(0, evenBytes);
                pcmDataBuffer = new Uint8Array(pcmDataBuffer.buffer.slice(evenBytes));
                playPCMChunk(chunkToProcess);
            }
        }
    };

    ws.onerror = () => {
        addToolLog("WebSocket", "action", "Connection error. Reconnecting...");
    };

    ws.onclose = (evt) => {
        const reason = evt.code === 1000 ? "normal close" : `code ${evt.code}`;
        addToolLog("WebSocket", "action", `Disconnected (${reason}). ${isCallActive ? "Reconnecting in 2s..." : ""}`);
        ws = null;
        if (isCallActive) {
            wsReconnectTimer = setTimeout(() => {
                if (isCallActive) connectWebSocket();
            }, 2000);
        }
    };
}

// ---------------------------------------------------------------------------
// Server Message Handlers
// ---------------------------------------------------------------------------
function handleServerMessage(message) {
    const event = message.event;

    if (event === "stream_start") {
        stopHoldMusic();
        isProcessing = false;
        const bubble = addChatBubble("assistant", "");
        currentAgentText = bubble.querySelector(".content p");
        
        // Fix "double voice": gracefully stop any playing audio from a previous turn or interruption
        if (audioPlayer) {
            audioPlayer.onended = null;
            audioPlayer.onerror = null;
            audioPlayer.pause();
        }
        audioQueue = [];
        isAudioPlaying = false;
        streamEnded = false;
        callStatusText.textContent = "Typing...";

    } else if (event === "text_chunk") {
        if (currentAgentText) {
            currentAgentText.textContent += message.text;
            transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
        }

    } else if (event === "audio_sentence") {
        playAgentAudio(message.audio_url);

    } else if (event === "citations") {
        if (message.citations) updateRAGUI(message.citations);

    } else if (event === "tags_update") {
        if (message.stage) {
            updatePipelineUI(message.stage);
            if (message.stage === "CLOSED") shouldDisconnect = true;
        }
        if (message.tags)  updateCRMUI(message.tags);

    } else if (event === "stream_end") {
        streamEnded = true;
        if (message.stage) {
            updatePipelineUI(message.stage);
            if (message.stage === "CLOSED") shouldDisconnect = true;
        }
        if (message.tags)  updateCRMUI(message.tags);
        
        // If TTS completely failed or text was empty
        if (!isAudioPlaying && audioQueue.length === 0) {
            finishAgentSpeaking();
        }

    } else if (event === "greeting" || event === "agent_response") {
        // Legacy fallback
        isProcessing = false;
        streamEnded = true; // Fix: ensure queue drains and opens mic after greeting
        addChatBubble("assistant", message.text);
        if (message.stage) updatePipelineUI(message.stage);
        if (message.tags)  updateCRMUI(message.tags);
        if (message.citations) updateRAGUI(message.citations);

        if (message.audio_url) {
            playAgentAudio(message.audio_url);
        } else {
            addToolLog("TTS", "action", "Text only (audio generation failed).");
            if (isCallActive) {
                callStatusText.textContent = "Listening...";
                startListening();
            }
        }

    } else if (event === "user_transcript") {
        addChatBubble("user", message.text);
        isProcessing = true;
        callStatusText.textContent = "Thinking...";
        playHoldMusic();

    } else if (event === "processing") {
        isProcessing = true;
        callStatusText.textContent = "Thinking...";
        playHoldMusic();

    } else if (event === "stt_error") {
        isProcessing = false;
        if (message.error) addToolLog("STT", "action", message.error);
        if (isCallActive) {
            callStatusText.textContent = "Listening...";
            startListening();
        }
    }
}

// ---------------------------------------------------------------------------
// Audio Playback — queues sentences and gaplessly plays them
// ---------------------------------------------------------------------------
let pcmContext = null;
let nextPlayTime = 0;

function interruptAudio() {
    isInterrupted = true;
    if (audioPlayer) {
        audioPlayer.pause();
    }
    // Also reset PCM playback
    if (pcmContext) {
        pcmContext.close();
        pcmContext = null;
    }
    nextPlayTime = 0;
    
    audioQueue = [];
    isAudioPlaying = false;
    isSpeaking = false;
    streamEnded = false;
    visualizerContainer.classList.remove("playing");
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ event: "interrupt" }));
    }
}

function playAgentAudio(url) {
    if (isInterrupted) return; // Prevent ghost audio from cancelled backend stream
    audioQueue.push(url);
    if (!isAudioPlaying) {
        playNextAudio();
    }
}

// ---------------------------------------------------------------------------
// Real-time PCM streaming playback (Sarvam WebSockets)
// ---------------------------------------------------------------------------
function playPCMChunk(buffer) {
    if (!pcmContext) {
        // Fallback if not initialized
        pcmContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    }
    
    // Resume context if suspended (browser policy)
    if (pcmContext.state === 'suspended') pcmContext.resume();
    
    // Reset nextPlayTime if it's lagging behind current time
    if (nextPlayTime < pcmContext.currentTime) {
        nextPlayTime = pcmContext.currentTime;
    }

    // Data is Int16 (2 bytes per sample) -> convert to Float32 [-1.0, 1.0]
    const int16Array = new Int16Array(buffer);
    const audioBuffer = pcmContext.createBuffer(1, int16Array.length, 24000);
    const float32Array = audioBuffer.getChannelData(0);
    
    for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
    }

    const source = pcmContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(pcmContext.destination);

    // Schedule playback seamlessly
    const startTime = Math.max(pcmContext.currentTime, nextPlayTime);
    source.start(startTime);
    nextPlayTime = startTime + audioBuffer.duration;

    isAudioPlaying = true;
    isSpeaking = true;
    callStatusText.textContent = "Speaking...";
    visualizerContainer.classList.add("playing");

    // When the queue is fully drained, clean up
    source.onended = () => {
        if (pcmContext && pcmContext.currentTime >= nextPlayTime - 0.05) {
            isAudioPlaying = false;
            isSpeaking = false;
            visualizerContainer.classList.remove("playing");
            if (isCallActive && !isProcessing) {
                callStatusText.textContent = "Listening...";
                // startListening(); // Removed to allow open mic barge-in
            }
        }
    };
}

// Legacy function for standard audio URLs
function playAgentAudio(url) {
    audioQueue.push(url);
    if (!isAudioPlaying) {
        playNextAudio();
    }
}

function playNextAudio() {
    if (audioQueue.length === 0) {
        if (streamEnded) {
            finishAgentSpeaking();
        }
        return;
    }

    const url = audioQueue.shift();
    isAudioPlaying = true;
    
    // We intentionally DO NOT stop listening here anymore, to allow Barge-in (Interruption).
    isSpeaking = true;
    callStatusText.textContent = "Speaking...";
    visualizerContainer.classList.add("playing");

    if (audioPlayer) {
        audioPlayer.onended = null;
        audioPlayer.onerror = null;
        audioPlayer.pause();
    }

    audioPlayer = new Audio(url);
    audioPlayer.onended = () => {
        isAudioPlaying = false;
        playNextAudio();
    };
    audioPlayer.onerror = () => {
        isAudioPlaying = false;
        playNextAudio();
    };
    audioPlayer.play().catch(() => {
        isAudioPlaying = false;
        playNextAudio();
    });
}

function finishAgentSpeaking() {
    isSpeaking = false;
    isAudioPlaying = false;
    streamEnded = false;
    visualizerContainer.classList.remove("playing");
    
    if (shouldDisconnect) {
        addToolLog("Call Engine", "action", "Conversation concluded. Automatically disconnecting call...");
        setTimeout(() => endCall(), 1000);
        shouldDisconnect = false;
        return;
    }
    
    if (isCallActive && !isProcessing) {
        callStatusText.textContent = "Listening...";
        startListening();
    }
}

// ---------------------------------------------------------------------------
// Hold Music (Web Audio API)
// ---------------------------------------------------------------------------
function playHoldMusic() {
    if (!holdAudioCtx) {
        holdAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (holdAudioCtx.state === 'suspended') holdAudioCtx.resume();
    
    stopHoldMusic();
    
    // Play a gentle double-chime every 2 seconds
    holdMusicInterval = setInterval(() => {
        if (!holdAudioCtx) return;
        
        const playChime = (delay, freq) => {
            const osc = holdAudioCtx.createOscillator();
            const gain = holdAudioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, holdAudioCtx.currentTime + delay);
            gain.gain.setValueAtTime(0, holdAudioCtx.currentTime + delay);
            gain.gain.linearRampToValueAtTime(0.05, holdAudioCtx.currentTime + delay + 0.1);
            gain.gain.exponentialRampToValueAtTime(0.001, holdAudioCtx.currentTime + delay + 0.5);
            osc.connect(gain);
            gain.connect(holdAudioCtx.destination);
            osc.start(holdAudioCtx.currentTime + delay);
            osc.stop(holdAudioCtx.currentTime + delay + 0.6);
        };
        
        playChime(0, 440); // First chime
        playChime(0.2, 554); // Second chime (major third)
        
    }, 2000);
}

function stopHoldMusic() {
    if (holdMusicInterval) {
        clearInterval(holdMusicInterval);
        holdMusicInterval = null;
    }
}

// ---------------------------------------------------------------------------
// Ambient Background Noise (HVAC/Room Tone simulation)
// ---------------------------------------------------------------------------
function startAmbientNoise() {
    if (!ambientAudioCtx) {
        ambientAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (ambientAudioCtx.state === 'suspended') ambientAudioCtx.resume();
    if (ambientNoiseNode) return; // already playing

    const bufferSize = 2 * ambientAudioCtx.sampleRate;
    const noiseBuffer = ambientAudioCtx.createBuffer(1, bufferSize, ambientAudioCtx.sampleRate);
    const output = noiseBuffer.getChannelData(0);
    
    // Generate Brown Noise (simulates HVAC / background hum)
    let lastOut = 0;
    for (let i = 0; i < bufferSize; i++) {
        let white = Math.random() * 2 - 1;
        output[i] = (lastOut + (0.02 * white)) / 1.02;
        lastOut = output[i];
        output[i] *= 3.5; 
    }

    ambientNoiseNode = ambientAudioCtx.createBufferSource();
    ambientNoiseNode.buffer = noiseBuffer;
    ambientNoiseNode.loop = true;

    // Filter to make it sound like distant room hum
    const filter = ambientAudioCtx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 350; // Muffled rumble

    ambientGain = ambientAudioCtx.createGain();
    ambientGain.gain.value = 0.08; // Very subtle hum

    ambientNoiseNode.connect(filter);
    filter.connect(ambientGain);
    ambientGain.connect(ambientAudioCtx.destination);
    
    ambientNoiseNode.start(0);
}

function stopAmbientNoise() {
    if (ambientNoiseNode) {
        try { ambientNoiseNode.stop(); } catch(e){}
        ambientNoiseNode.disconnect();
        ambientNoiseNode = null;
    }
}

// ---------------------------------------------------------------------------
// Input Dispatch
// ---------------------------------------------------------------------------
function startListening() {
    if (!isCallActive || isSpeaking || isProcessing) return;

    const mode = inputModeSelect.value;
    const vk   = languageSelect.value;

    if (mode === "speech_api") {
        startWebSpeechRecognition(vk);
    } else {
        startMicRecording(vk);
    }
}

function stopListening() {
    // Stop Web Speech API
    if (speechRecognizer) {
        try { speechRecognizer.abort(); } catch(e) {}
        speechRecognizer = null;
    }
    // Stop Whisper mic recording
    if (vadInterval) { clearInterval(vadInterval); vadInterval = null; }
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        try { mediaRecorder.stop(); } catch(e) {}
    }
    mediaRecorder = null;
    audioChunks   = [];
    if (userMediaStream) {
        userMediaStream.getTracks().forEach(t => t.stop());
        userMediaStream = null;
    }
    if (audioContext && audioContext.state !== "closed") {
        try { audioContext.close(); } catch(e) {}
        audioContext = null;
    }
    analyser = null;
}

// ---------------------------------------------------------------------------
// PRIMARY INPUT: Web Speech API (continuous, dynamic language)
// Chrome supports: en-IN, hi-IN, te-IN natively — zero latency, no upload.
// ---------------------------------------------------------------------------
function startWebSpeechRecognition(voiceKey) {
    // Don't create a second instance if one is already running
    if (speechRecognizer) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        addToolLog("Microphone", "action", "Web Speech API not available in this browser. Using Whisper fallback.");
        inputModeSelect.value = "mic_stream";
        startMicRecording(voiceKey);
        return;
    }

    const langCode = getLangCode(voiceKey);
    const recognizer = new SpeechRecognition();

    // continuous=true: keeps listening between utterances without manual restart
    recognizer.continuous      = true;
    recognizer.interimResults  = true;   // Show interim for responsiveness
    recognizer.lang            = langCode;
    recognizer.maxAlternatives = 1;

    recognizer.onstart = () => {
        addToolLog("Microphone", "success", `Listening (${langCode}) — speak now...`);
        callStatusText.textContent = "Listening...";
    };

    let speechSilenceTimer = null;

    // Show interim results in the status bar for visual feedback
    recognizer.onresult = (event) => {
        if (!isCallActive || isProcessing) return;

        let interimTranscript = "";
        let finalTranscript   = "";

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const t = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += t;
            } else {
                interimTranscript += t;
            }
        }

        // Show interim in status bar (live typing feel)
        if (interimTranscript) {
            callStatusText.textContent = `"${interimTranscript.slice(0, 40)}..."`;
            
            // ULTRA LOW LATENCY HACK: Chrome natively waits ~2 seconds to finalize.
            // We force it to finalize after 700ms of no new words.
            if (speechSilenceTimer) clearTimeout(speechSilenceTimer);
            speechSilenceTimer = setTimeout(() => {
                if (speechRecognizer && interimTranscript.trim().length > 2) {
                    addToolLog("Microphone", "success", "Fast silence detected. Forcing flush.");
                    try { speechRecognizer.stop(); } catch(e) {}
                }
            }, 700);
        }

        // Only send final transcripts to the agent
        if (finalTranscript.trim()) {
            if (speechSilenceTimer) {
                clearTimeout(speechSilenceTimer);
                speechSilenceTimer = null;
            }
            
            // Prevent ECHO loop: Only interrupt for intentional phrases, not tiny blips of the bot's own voice
            if (isSpeaking) {
                if (finalTranscript.trim().length < 8) {
                    addToolLog("Interruption", "warning", `Ignored short phrase to prevent echo: "${finalTranscript.trim()}"`);
                    return;
                }
                interruptAudio();
            }
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ event: "text_input", text: finalTranscript.trim(), voice_key: voiceKey }));
            }
        }
    };

    recognizer.onerror = (e) => {
        console.warn("Speech error:", e.error);
        if (e.error === "not-allowed" || e.error === "service-not-allowed") {
            // Permanent — mic permission denied
            addToolLog("Microphone", "action",
                "⚠️ Mic permission DENIED. Click the 🔒 icon in the address bar → allow microphone → refresh the page.");
            callStatusText.textContent = "Mic blocked";
            speechRecognizer = null;
            return;
        }
        // network/audio-capture/no-speech are transient — recognizer will auto-end and restart
        speechRecognizer = null;
    };

    recognizer.onend = () => {
        speechRecognizer = null;
        // Auto-restart unless the bot is speaking or call ended
        if (isCallActive && !isSpeaking && !isProcessing) {
            setTimeout(() => {
                if (isCallActive && !isSpeaking && !isProcessing && !speechRecognizer) {
                    startWebSpeechRecognition(voiceKey);
                }
            }, 300);
        }
    };

    try {
        recognizer.start();
        speechRecognizer = recognizer;
    } catch (e) {
        console.warn("Could not start recognition:", e);
        speechRecognizer = null;
    }
}

// ---------------------------------------------------------------------------
// FALLBACK INPUT: Whisper via mic upload with VAD silence detection
// Use when: (a) Web Speech API unavailable, (b) user manually selects Whisper
// ---------------------------------------------------------------------------
async function startMicRecording(voiceKey) {
    if (mediaRecorder && mediaRecorder.state === "recording") return;
    stopListening();

    try {
        userMediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            },
            video: false
        });
    } catch (err) {
        let msg = "Mic access denied.";
        if (err.name === "NotAllowedError") {
            msg = "⚠️ Mic permission DENIED. Click the 🔒 icon in the address bar → allow microphone → refresh.";
        } else if (err.name === "NotFoundError") {
            msg = "No microphone found. Please connect a mic and retry.";
        }
        addToolLog("Microphone", "action", msg);
        callStatusText.textContent = "Mic blocked";
        return;
    }

    // Pick the best available MIME type
    let mimeType = "audio/webm;codecs=opus";
    if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = "audio/webm";
    if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = "audio/ogg;codecs=opus";
    if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = "audio/ogg";
    if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = "audio/mp4"; // iOS Safari
    if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = ""; // Browser default

    mediaRecorder = new MediaRecorder(userMediaStream, { mimeType });
    audioChunks   = [];

    mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
        if (!isCallActive || audioChunks.length === 0) { audioChunks = []; return; }

        const blob = new Blob(audioChunks, { type: mimeType });
        audioChunks = [];

        // Skip tiny blobs (noise / accidental trigger)
        if (blob.size < 100) {
            if (isCallActive && !isSpeaking && !isProcessing) {
                setTimeout(() => startMicRecording(voiceKey), 300);
            }
            return;
        }

        addToolLog("Whisper", "running", `Sending audio blob (${blob.size} bytes) to server...`);

        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => {
            const b64 = reader.result.split(",")[1];

            // Set language hint — or null for auto-detect across all Indian languages
            let langHint = null;  // Auto-detect: Whisper identifies the language itself
            if (voiceKey.startsWith("te")) langHint = "te";
            else if (voiceKey.startsWith("hi")) langHint = "hi";
            else if (voiceKey.startsWith("en")) langHint = "en";

            // Determine extension for Whisper MIME detection
            const ext  = mimeType.includes("webm") ? "webm" : "ogg";
            const fname = `audio.${ext}`;

            if (isSpeaking) interruptAudio();

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    event: "audio_input",
                    audio: b64,
                    voice_key: voiceKey,
                    filename: fname
                }));
            }
            // Mic stays closed — will restart in finishAgentSpeaking after bot responds
        };
    };

    // VAD: RMS energy analysis via AudioContext
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(userMediaStream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);

    const dataArray    = new Uint8Array(analyser.frequencyBinCount);
    let hasSpoken      = false;
    let silenceStart   = null;
    const THRESH       = 0.002;     // EXTREMELY low threshold to catch even the quietest microphones!
    const SILENCE_MS   = 800;       // Wait slightly longer to prevent cutting off words
    const MAX_WAIT_MS  = 12000;     // 12s idle → restart
    const waitStart    = Date.now();

    vadInterval = setInterval(() => {
        if (!isCallActive || isProcessing) {
            clearInterval(vadInterval); vadInterval = null; return;
        }
        analyser.getByteTimeDomainData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            const v = (dataArray[i] - 128) / 128;
            sum += v * v;
        }
        const rms = Math.sqrt(sum / dataArray.length);

        if (rms > THRESH) {
            hasSpoken    = true;
            silenceStart = null;
            callStatusText.textContent = "Recording...";
            if (isSpeaking) interruptAudio();
        } else if (hasSpoken) {
            if (!silenceStart) { silenceStart = Date.now(); return; }
            if (Date.now() - silenceStart > SILENCE_MS) {
                clearInterval(vadInterval); vadInterval = null;
                addToolLog("VAD", "success", "Speech captured — processing...");
                if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
            }
        } else {
            // Still waiting for first word
            if (Date.now() - waitStart > MAX_WAIT_MS) {
                clearInterval(vadInterval); vadInterval = null;
                addToolLog("VAD", "action", "No speech detected — restarting mic.");
                if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
                setTimeout(() => { if (isCallActive && !isSpeaking) startMicRecording(voiceKey); }, 500);
            }
        }
    }, 50);

    mediaRecorder.start(200);
    addToolLog("Microphone", "success", "Recording — speak now...");
    callStatusText.textContent = "Listening...";
}

// ---------------------------------------------------------------------------
// Keyboard Input
// ---------------------------------------------------------------------------
function sendKeyboardInput() {
    const text = keyboardInput.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(jsonPacket("text_input", { text, voice_key: languageSelect.value }));
    keyboardInput.value = "";
}

// ---------------------------------------------------------------------------
// UI Renderers
// ---------------------------------------------------------------------------
function addChatBubble(role, text) {
    const bubble = document.createElement("div");
    bubble.classList.add("chat-bubble", role);
    const icon = role === "user" ? "fa-user" : "fa-robot";
    bubble.innerHTML = `
        <div class="avatar"><i class="fa-solid ${icon}"></i></div>
        <div class="content"><p>${escapeHtml(text)}</p></div>
    `;
    bubble.style.opacity = "0";
    bubble.style.transform = "translateY(8px)";
    transcriptContainer.appendChild(bubble);
    transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
    requestAnimationFrame(() => {
        bubble.style.transition = "all 0.25s ease";
        bubble.style.opacity = "1";
        bubble.style.transform = "translateY(0)";
    });
    return bubble;
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function updatePipelineUI(activeStage) {
    const stages = ["G", "I", "Q", "PE", "PR", "C"];
    const idx    = stages.indexOf(activeStage);
    stages.forEach((s, i) => {
        const el = document.getElementById(`step-${s}`);
        if (!el) return;
        el.classList.remove("active", "completed");
        if (i === idx) el.classList.add("active");
        else if (i < idx) el.classList.add("completed");
    });
}

function updateCRMUI(tags) {
    crmTags.callerName.textContent = tags.caller_name || "Unsure";
    toggleBadgeClass(crmTags.callerName, tags.caller_name, "badge-active");

    crmTags.relation.textContent = tags.patient_relation || "Unsure";
    toggleBadgeClass(crmTags.relation, tags.patient_relation, "badge-active");

    crmTags.cancer.textContent = tags.cancer_type || "Unsure";
    toggleBadgeClass(crmTags.cancer, tags.cancer_type, "badge-active");

    const syms = tags.symptoms || [];
    crmTags.symptoms.textContent = syms.length ? syms.join(", ") : "None";
    crmTags.symptoms.title = syms.join(", ");

    crmTags.authority.textContent = tags.decision_authority || "Unsure";
    toggleBadgeClass(crmTags.authority, tags.decision_authority, "badge-active");

    crmTags.travel.textContent = tags.travel_fit || "Unsure";
    crmTags.travel.className = "tag-value badge";
    if (tags.travel_fit === "Yes") crmTags.travel.classList.add("badge-yes");
    else if (tags.travel_fit === "No") crmTags.travel.classList.add("badge-no");

    crmTags.temperature.textContent = tags.lead_temperature || "Warm";
    crmTags.temperature.className = "tag-value badge";
    const temp = (tags.lead_temperature || "").toLowerCase();
    if (temp === "hot") crmTags.temperature.classList.add("badge-hot");
    else if (temp === "warm") crmTags.temperature.classList.add("badge-warm");
    else if (temp === "cold") crmTags.temperature.classList.add("badge-cold");

    crmTags.persona.textContent = tags.buyer_persona || "Unsure";
    toggleBadgeClass(crmTags.persona, tags.buyer_persona, "badge-active");

    const booking = tags.consultation_booked || "None";
    crmTags.booking.textContent = booking;
    crmTags.booking.className = "tag-value badge";
    if (booking === "Premium") {
        crmTags.booking.classList.add("badge-premium");
        addToolLog("CRM", "success", "PREMIUM booked ₹5,000 — payment link sent.");
    } else if (booking === "Standard") {
        crmTags.booking.classList.add("badge-standard");
        addToolLog("CRM", "success", "STANDARD booked ₹3,000 — payment link sent.");
    }
    addToolLog("CRM", "success", "Lead tags updated.");
}

function updateRAGUI(citations) {
    if (!citations || !citations.length) return;
    ragContainer.innerHTML = "";
    citations.forEach(doc => {
        const card = document.createElement("div");
        card.classList.add("rag-citation-card");
        card.innerHTML = `<h5><i class="fa-solid fa-file-invoice"></i> ${escapeHtml(doc.title)}</h5><p>${escapeHtml(doc.text)}</p>`;
        ragContainer.appendChild(card);
    });
    addToolLog("RAG", "success", `${citations.length} knowledge citations retrieved.`);
}

function toggleBadgeClass(el, val, cls) {
    if (val && val !== "Unsure" && val !== "None") el.classList.add(cls);
    else el.classList.remove(cls);
}

function addToolLog(source, type, details) {
    const item = document.createElement("div");
    item.classList.add("tool-log-item");
    const t = new Date().toTimeString().split(" ")[0];
    const icon = type === "success" ? "fa-check" : type === "running" ? "fa-spinner fa-spin" : "fa-info";
    item.innerHTML = `
        <span class="time">${t}</span>
        <span class="status ${type}"><i class="fa-solid ${icon}"></i> ${escapeHtml(source)}</span>
        <span class="details">${escapeHtml(details)}</span>
    `;
    toolsLog.appendChild(item);
    toolsLog.scrollTop = toolsLog.scrollHeight;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function jsonPacket(event, data) {
    return JSON.stringify({ event, ...data });
}
