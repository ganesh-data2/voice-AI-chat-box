import streamlit as st
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

from answer_engine import AnswerEngine
from speech_utils import tts_play, stt_transcribe_numpy, tts_play_to_bytes

import tempfile
from pydub import AudioSegment
from pydub.playback import play

# 📘 Load the FAQ PDF
engine = AnswerEngine("ilovepdf_merged.pdf")

# 🎙️ Microphone audio processor using recv_queued
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_queued(self, frames):
        for frame in frames:
            audio = frame.to_ndarray()
            self.frames.append(audio)
        return frames[-1] if frames else None

    def get_audio(self):
        if not self.frames:
            return None
        return np.concatenate(self.frames, axis=1).flatten().astype(np.float32)

# 🔊 Function to play base64-encoded MP3 audio
def play_base64_audio(b64_audio):
    import base64
    decoded = base64.b64decode(b64_audio)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio.write(decoded)
    temp_audio.flush()
    audio = AudioSegment.from_file(temp_audio.name, format="mp3")
    play(audio)

# 🧠 Ask the AI with transcript & voice
def ask_ai(query):
    st.session_state.chat_history.append(("user", query))
    st.session_state.transcript += f"👤 You: {query}\n"

    with st.chat_message("user"):
        st.markdown(query)

    # 🔊 Speak the user's input aloud
    st.info("🔊 Speaking your question...")
    tts_play(query)

    answer = engine.get_answer(query)

    # Out-of-scope handling
    if not answer or "no relevant" in answer.lower() or "not found" in answer.lower():
        answer = "I do not have an answer to that question, let me transfer your call to the live agent."
        with st.expander("🚨 Live Agent Required"):
            st.warning("This query could not be answered from the document. Please route to a live agent.")

    st.session_state.chat_history.append(("assistant", answer))
    st.session_state.transcript += f"🤖 Bot: {answer}\n\n"

    with st.chat_message("assistant"):
        st.markdown(answer)

    # 🔊 Speak the bot's answer
    st.info("🔊 Speaking the answer...")
    tts_play(answer)

# 🚀 Streamlit app config
st.set_page_config(page_title="Voice FAQ Bot", layout="centered")
st.title("🎙️ Offline Voice FAQ Chatbot")

# 🔄 Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# 🎧 Setup mic with WebRTC
ctx = webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=2048,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# 🎤 Record and process voice
if ctx.state.playing and ctx.audio_processor and st.button("🎤 Speak Now"):
    raw_audio = ctx.audio_processor.get_audio()

    if raw_audio is None or len(raw_audio) < 2500:
        st.warning("⚠️ No or very short audio detected. Please speak clearly and try again.")
    else:
        st.write(f"🎧 Audio sample length: {len(raw_audio)}")
        st.info("🔄 Transcribing...")

        transcript = stt_transcribe_numpy(raw_audio, sample_rate=48000)

        if transcript:
            st.success(f"🗣️ You said: **{transcript}**")
            ask_ai(transcript)
        else:
            st.error("❌ Sorry, I couldn't understand your voice. Try again more clearly.")

# 💬 Text fallback: now speaks user's text + bot response
typed = st.chat_input("Or type your question here...")
if typed:
    ask_ai(typed)

# 📜 Transcript
with st.expander("📄 Transcript"):
    st.text_area("Conversation Transcript", st.session_state.transcript, height=300)
