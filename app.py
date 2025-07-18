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
    """Loads and initializes the Moshi and Mimi models the correct way."""
    global mimi, lm_gen, text_tokenizer
    if mimi and lm_gen:
        logger.info("Models already loaded.")
        return

    try:
        logger.info("Loading Moshi models from Hugging Face...")
        
        # Define the repository to use
        repo_id = "kyutai/moshiko-pytorch-bf16" 
        
        # ---
        # FINAL VERIFIED FIX: The 'repo_id' argument must be positional, not a keyword.
        # This is the correct way to call the function based on the library's source code.
        # ---
        logger.info(f"Fetching model info from repo: {repo_id}")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(repo_id)
        logger.info("Checkpoint info retrieved successfully.")

        # Now, load the models from the information the library gathered.
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
    # Load the ML models on startup
    initialize_models()
    yield
    # Clean up on shutdown
    global mimi, lm_gen, text_tokenizer
    mimi, lm_gen, text_tokenizer = None, None, None
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
    
    opus_reader = sphn.OpusStreamReader(sample_rate=mimi.sample_rate)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    try:
        with torch.no_grad(), lm_gen.streaming(1), mimi.streaming(1):
            while True:
                message = await websocket.receive()

                if "bytes" in message:
                    opus_reader.append_bytes(message["bytes"])
                elif "text" in message:
                    data = json.loads(message["text"])
                    if data.get("action") == "set_emotion":
                        state.set_emotion(data.get("emotion"))
                    continue

                audio_frames = opus_reader.read_pcm()
                if audio_frames.shape[-1] < frame_size:
                    continue 

                num_frames = audio_frames.shape[-1] // frame_size
                for i in range(num_frames):
                    start = i * frame_size
                    end = start + frame_size
                    chunk = torch.from_numpy(audio_frames[..., start:end]).to(device).unsqueeze(0)

                    codes = mimi.encode(chunk)
                    tokens_out = lm_gen.step(codes)
                    
                    if tokens_out is not None:
                        audio_out_pcm = mimi.decode(tokens_out[:, 1:]).cpu().numpy().squeeze()
                        await websocket.send_bytes(audio_out_pcm.astype(np.float32).tobytes())

                        text_token = tokens_out[0, 0, 0].item()
                        if text_token not in (0, 3, text_tokenizer.eos_id(), text_tokenizer.pad_id()):
                            ai_text = text_tokenizer.decode([text_token])
                            response = {
                                "type": "transcript",
                                "data": {"ai_text": state.apply_emotion_to_text(ai_text)}
                            }
                            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"An error occurred in the WebSocket handler: {e}", exc_info=True)
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed.")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
