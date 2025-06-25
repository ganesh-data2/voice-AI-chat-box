from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import io
import wave
from answer_engine import AnswerEngine
from speech_utils import stt_transcribe_numpy, tts_play_to_bytes

app = FastAPI()
engine = AnswerEngine("ilovepdf_merged.pdf")

@app.post("/ask_audio/")
async def ask_audio(file: UploadFile = File(...)):
    # Step 1: Read WAV audio and convert to NumPy array
    wav_bytes = await file.read()
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    # Step 2: STT
    transcript = stt_transcribe_numpy(audio, sample_rate=sample_rate)

    if not transcript:
        return JSONResponse(status_code=400, content={"error": "Could not transcribe audio"})

    # Step 3: FAQ Answer
    answer = engine.get_answer(transcript)
    if not answer or "no relevant" in answer.lower():
        answer = "I do not have an answer to that. Please contact a human agent."

    # Step 4: TTS
    audio_bytes = tts_play_to_bytes(answer)  # You must define this in speech_utils.py

    return {
        "transcript": transcript,
        "answer": answer,
        "tts_audio_base64": audio_bytes
    }
