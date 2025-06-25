import speech_recognition as sr
import uuid
import os
import soundfile as sf
import numpy as np
import base64
import io
from gtts import gTTS

# ✅ Web-compatible TTS using gTTS and base64-encoded MP3 (Streamlit-friendly)
def tts_play(text: str) -> str:
    try:
        tts = gTTS(text)
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        b64_audio = base64.b64encode(audio_io.read()).decode("utf-8")
        return f"data:audio/mp3;base64,{b64_audio}"
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return None

# ✅ STT from NumPy audio array using Google Speech Recognition
def stt_transcribe_numpy(audio_array: np.ndarray, sample_rate: int = 48000) -> str:
    filename = f"temp_{uuid.uuid4().hex}.wav"
    try:
        # Save audio as WAV file
        sf.write(filename, audio_array, sample_rate)

        # Transcribe using recognizer
        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        print(f"[DEBUG STT] Transcribed text: {text}")
        return text.strip()

    except sr.UnknownValueError:
        print("[DEBUG STT] Could not understand the audio.")
        return ""
    except Exception as e:
        print(f"[DEBUG STT] Error: {e}")
        return ""
    finally:
        if os.path.exists(filename):
            os.remove(filename)
