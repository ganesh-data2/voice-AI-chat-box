import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from answer_engine import AnswerEngine
from speech_utils import tts_play, stt_transcribe_numpy
import base64

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

# 🧠 Ask the AI with transcript & voice
def ask_ai(query):
    st.session_state.chat_history.append(("user", query))
    st.session_state.transcript += f"👤 You: {query}\n"

    with st.chat_message("user"):
        st.markdown(query)

    # 🔊 Speak the user's question (TTS)
    audio_data = tts_play(query)
    if audio_data:
        st.audio(audio_data, format="audio/mp3")

    # 🤖 Get answer from FAQ
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

    # 🔊 Speak the bot's response (TTS)
    audio_data = tts_play(answer)
    if audio_data:
        st.audio(audio_data, format="audio/mp3")

# 🚀 Streamlit app config
st.set_page_config(page_title="Voice FAQ Bot", layout="centered")
st.title("🎙️ Voice FAQ Chatbot (Web Compatible)")

# 🔄 Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# 🎧 Setup mic with WebRTC (will not work on Streamlit Cloud but retained for local use)
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

# 💬 Text fallback input: now speaks both user's text + AI response
typed = st.chat_input("Or type your question here...")

if typed:
    # Speak user input
    user_audio = speak_text(typed)
    st.audio(user_audio, format="audio/mp3")

    # Get AI answer
    answer = ask_ai(typed)

    # Speak AI response
    bot_audio = speak_text(answer)
    st.audio(bot_audio, format="audio/mp3")

# 📜 Transcript display
with st.expander("📄 Transcript"):
    st.text_area("Conversation Transcript", st.session_state.transcript, height=300)

