document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const startBtn = document.getElementById('startCall');
    const endBtn = document.getElementById('endCall');
    const muteBtn = document.getElementById('muteBtn');
    const statusEl = document.getElementById('connectionStatus');
    const currentEmotionEl = document.getElementById('currentEmotion');
    const userTranscriptEl = document.getElementById('userTranscriptText');
    const aiTranscriptEl = document.getElementById('aiTranscriptText');
    const emotionButtonsContainer = document.querySelector('.emotion-buttons');

    // --- State Variables ---
    let ws = null;
    let mediaRecorder = null;
    let audioContext = null;
    let audioSource = null;
    let isMuted = false;
    let isConnected = false;
    let currentEmotion = 'happy';
    let audioQueue = [];
    let isPlaying = false;

    // --- Event Listeners ---
    startBtn.addEventListener('click', startCall);
    endBtn.addEventListener('click', endCall);
    muteBtn.addEventListener('click', toggleMute);
    emotionButtonsContainer.addEventListener('click', (e) => {
        if (e.target.classList.contains('emotion-btn')) {
            const emotion = e.target.dataset.emotion;
            setEmotion(emotion);
        }
    });

    // --- Core Functions ---
    async function startCall() {
        if (isConnected) return;
        console.log("Attempting to start call...");

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)();

            const wsUrl = `ws://${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);
            ws.binaryType = "arraybuffer"; // Important for audio data

            ws.onopen = () => {
                console.log("WebSocket connected.");
                isConnected = true;
                updateUIForConnection(true);
                startRecording(stream);
            };

            ws.onmessage = (event) => {
                if (event.data instanceof ArrayBuffer) {
                    // Received raw audio bytes
                    const audioData = new Float32Array(event.data);
                    audioQueue.push(audioData);
                    playFromQueue();
                } else if (typeof event.data === 'string') {
                    // Received JSON transcript/metadata
                    const message = JSON.parse(event.data);
                    handleServerMessage(message);
                }
            };

            ws.onclose = () => {
                console.log("WebSocket disconnected.");
                isConnected = false;
                updateUIForConnection(false);
                stopRecording();
            };

            ws.onerror = (error) => {
                console.error("WebSocket Error:", error);
                alert("Connection failed. Check the server and console logs.");
                endCall();
            };

        } catch (error) {
            console.error("Could not start call:", error);
            alert("Microphone access was denied. Please allow microphone access in your browser settings.");
        }
    }

    function endCall() {
        if (ws) {
            ws.close();
        }
        stopRecording();
        updateUIForConnection(false);
        audioQueue = []; // Clear any pending audio
    }

    function startRecording(stream) {
        if (mediaRecorder && mediaRecorder.state === 'recording') return;
        
        // Using Opus codec if available for better performance
        const options = { mimeType: 'audio/webm;codecs=opus' };
        mediaRecorder = new MediaRecorder(stream, options);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0 && ws && ws.readyState === WebSocket.OPEN && !isMuted) {
                ws.send(event.data);
            }
        };

        mediaRecorder.start(250); // Send audio data every 250ms
        console.log("MediaRecorder started.");
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            console.log("MediaRecorder stopped.");
        }
    }

    function toggleMute() {
        isMuted = !isMuted;
        muteBtn.textContent = isMuted ? 'Unmute' : 'Mute';
        muteBtn.style.background = isMuted ? '#ff9800' : '#2196F3';
    }

    function setEmotion(emotion) {
        if (!isConnected) {
            alert("You must start a call to change emotions.");
            return;
        }
        currentEmotion = emotion;
        currentEmotionEl.textContent = emotion;
        updateEmotionButtons();
        
        // Send emotion change to the server
        const command = { action: 'set_emotion', emotion: emotion };
        ws.send(JSON.stringify(command));
        console.log(`Sent emotion change to server: ${emotion}`);
    }

    function handleServerMessage(message) {
        if (message.type === 'transcript') {
            const { ai_text } = message.data;
            if (ai_text) {
                aiTranscriptEl.textContent = ai_text;
            }
        }
    }

    // --- Audio Playback ---
    function playFromQueue() {
        if (isPlaying || audioQueue.length === 0) {
            return;
        }
        isPlaying = true;
        
        const audioData = audioQueue.shift();
        const audioBuffer = audioContext.createBuffer(1, audioData.length, audioContext.sampleRate);
        audioBuffer.copyToChannel(audioData, 0);

        audioSource = audioContext.createBufferSource();
        audioSource.buffer = audioBuffer;
        audioSource.connect(audioContext.destination);
        
        audioSource.onended = () => {
            isPlaying = false;
            playFromQueue(); // Play next chunk if available
        };

        audioSource.start();
    }

    // --- UI Updates ---
    function updateUIForConnection(connected) {
        startBtn.disabled = connected;
        endBtn.disabled = !connected;
        muteBtn.disabled = !connected;
        statusEl.textContent = connected ? 'Connected' : 'Disconnected';
        statusEl.className = connected ? 'status-indicator connected' : 'status-indicator disconnected';
    }

    function updateEmotionButtons() {
        document.querySelectorAll('.emotion-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.emotion === currentEmotion);
        });
    }

    updateUIForConnection(false); // Initial state
});
