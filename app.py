import asyncio
import json
import logging
from contextlib import asynccontextmanager
import torch
import numpy as np
import sphn
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from huggingface_hub import hf_hub_download
from moshi.models import loaders, LMGen

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Model Variables ---
mimi = None
lm_gen = None
text_tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Loading ---
def initialize_models():
    """Loads and initializes the Moshi and Mimi models."""
    global mimi, lm_gen, text_tokenizer
    if mimi and lm_gen:
        logger.info("Models already loaded.")
        return

    try:
        logger.info("Loading Moshi models from Hugging Face...")
        
        # Using the official Kyutai repository for Moshiko (male voice)
        repo_id = "kyutai/moshiko-pytorch-bf16" 
        
        mimi_path = hf_hub_download(repo=repo_id, filename="mimi.bin")
        moshi_path = hf_hub_download(repo=repo_id, filename="moshi.bin")
        tokenizer_path = hf_hub_download(repo=repo_id, filename="tokenizer.model")

        checkpoint_info = loaders.CheckpointInfo(
            repo_id, moshi_path, mimi_path, tokenizer_path
        )

        mimi = checkpoint_info.get_mimi(device=device)
        mimi.set_num_codebooks(8)
        
        moshi_lm = checkpoint_info.get_moshi(device=device)
        text_tokenizer = checkpoint_info.get_text_tokenizer()
        
        # LMGen handles the generation logic
        lm_gen = LMGen(moshi_lm, temp=0.8, temp_text=0.7)
        
        logger.info(f"Models loaded successfully on device: {device}")
    except Exception as e:
        logger.error(f"Fatal error loading models: {e}", exc_info=True)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models
    initialize_models()
    yield
    # Clean up the ML models and release the resources
    global mimi, lm_gen
    mimi = None
    lm_gen = None
    logger.info("Cleaned up models.")


# --- FastAPI App Initialization ---
app = FastAPI(title="MoshiAI Voice Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- Emotion and State Management ---
class ConversationState:
    def __init__(self):
        self.current_emotion = "happy"

    def set_emotion(self, emotion: str):
        self.current_emotion = emotion
        logger.info(f"Emotion set to: {self.current_emotion}")

    def apply_emotion_to_text(self, text: str) -> str:
        """Applies emotional context to the AI's response text."""
        # This is a placeholder for more advanced emotional TTS.
        # Moshi's fine-tuning would be the proper way to achieve this.
        # For now, we prepend a tag that the user can see.
        tags = {
            "whispering": "*whispers*",
            "giggling": "*giggles*",
            "sad": "*sighs*",
            "dramatic": "*dramatically*"
        }
        tag = tags.get(self.current_emotion, "")
        return f"{tag} {text}".strip()


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the real-time bidirectional audio stream."""
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    
    state = ConversationState()
    
    # Initialize streaming components for this connection
    opus_reader = sphn.OpusStreamReader(sample_rate=mimi.sample_rate)
    opus_writer = sphn.OpusStreamWriter(sample_rate=mimi.sample_rate)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    async def forward_to_client():
        """Listens for generated audio from the model and sends it to the client."""
        while True:
            try:
                # Get generated audio bytes from the writer
                audio_bytes = opus_writer.read_bytes()
                if len(audio_bytes) > 0:
                    await websocket.send_bytes(audio_bytes)
                await asyncio.sleep(0.01) # Non-blocking sleep
            except WebSocketDisconnect:
                logger.info("Client disconnected during send.")
                break
            except Exception as e:
                logger.error(f"Error in send loop: {e}")
                break

    client_sender_task = asyncio.create_task(forward_to_client())

    try:
        with torch.no_grad(), lm_gen.streaming(1), mimi.streaming(1):
            while True:
                # 1. Receive data from client (could be audio or JSON command)
                message = await websocket.receive()

                if "bytes" in message:
                    # Append incoming audio bytes to the Opus reader
                    opus_reader.append_bytes(message["bytes"])
                elif "text" in message:
                    # Handle JSON commands from the client (e.g., changing emotion)
                    data = json.loads(message["text"])
                    if data.get("action") == "set_emotion":
                        state.set_emotion(data.get("emotion"))
                        continue

                # 2. Process complete audio frames
                audio_frames = opus_reader.read_pcm()
                if audio_frames.shape[-1] < frame_size:
                    continue # Not enough data for a full frame

                num_frames = audio_frames.shape[-1] // frame_size
                for i in range(num_frames):
                    start = i * frame_size
                    end = start + frame_size
                    chunk = torch.from_numpy(audio_frames[..., start:end]).to(device).unsqueeze(0)

                    # 3. Encode user audio with Mimi
                    codes = mimi.encode(chunk)
                    
                    # 4. Generate response with Moshi LM
                    tokens_out = lm_gen.step(codes)
                    
                    if tokens_out is not None:
                        # Decode AI audio response
                        audio_out = mimi.decode(tokens_out[:, 1:]).cpu().numpy().squeeze()
                        opus_writer.append_pcm(audio_out)

                        # Decode AI text response for transcription
                        text_token = tokens_out[0, 0, 0].item()
                        if text_token not in (0, 3, text_tokenizer.eos_id(), text_tokenizer.pad_id()):
                            ai_text = text_tokenizer.decode([text_token])
                            # Send transcription to client
                            response = {
                                "type": "transcript",
                                "data": {
                                    "ai_text": state.apply_emotion_to_text(ai_text)
                                }
                            }
                            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"An error occurred in the WebSocket handler: {e}", exc_info=True)
    finally:
        client_sender_task.cancel()
        await websocket.close()
        logger.info("WebSocket connection closed.")


if __name__ == "__main__":
    # Ensure models are loaded before starting the server if running directly
    if not mimi or not lm_gen:
        initialize_models()
    
    # Use reload=True for development to auto-restart on code changes
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
