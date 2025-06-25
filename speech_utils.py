import pyttsx3
import speech_recognition as sr
import uuid
import os
import soundfile as sf
import numpy as np
import base64
import io
from gtts import gTTS

# ✅ Offline TTS using pyttsx3 (plays directly on speakers)
def tts_play(text: str):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"❌ TTS Error: {e}")

# ✅ STT from NumPy audio array (wav) using Google recognizer
def stt_transcribe_numpy(audio_array: np.ndarray, sample_rate: int = 48000) -> str:
    filename = f"temp_{uuid.uuid4().hex}.wav"
    try:
        # Save NumPy audio to WAV
        sf.write(filename, audio_array, sample_rate)

        # Transcribe using Google STT (online)
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

# ✅ Optional: TTS that returns base64 MP3 (for future use or saving)
def tts_play_to_bytes(text: str) -> str:
    try:
        tts = gTTS(text)
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        return base64.b64encode(audio_io.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ TTS encoding error: {e}")
        return ""
