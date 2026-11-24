import os
import io
import wave
import threading
import requests
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

load_dotenv()

# --- KONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:4b"
WHISPER_SIZE = "small"  # "base" ist noch schneller, "small" versteht Akzente deutlich besser
SAMPLE_RATE = 16000

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
# Standard-Stimme "Daniel" oder "Aria" (funktioniert sehr gut auf Deutsch mit multilingual/flash v2.5)
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "onwK4e9ZLuTAKqWW03F9")

MODES = {
    "1": {
        "name": "Tech-Interview (Linz AI / Backend)",
        "prompt": (
            "Du bist ein freundlicher, aber anspruchsvoller Senior Software & AI Engineer bei einem Tech-Unternehmen in Linz (Österreich). "
            "Du führst ein lockeres technisches Kennenlerngespräch auf Deutsch (B2-Niveau). "
            "Stelle gezielte Fragen zu Python, Daten-Pipelines, APIs oder KI-Projekten. "
            "WICHTIG: Antworte IMMER kurz (maximal 2-3 gesprochene Sätze), natürlich und ohne Markdown, Sternchen oder Aufzählungen, "
            "da dein Text direkt vorgelesen wird. Stelle am Ende immer genau eine Anschlussfrage."
        ),
    },
    "2": {
        "name": "Daily Stand-up im Entwicklerteam",
        "prompt": (
            "Du bist Tech Lead in einem österreichischen Software-Team in Linz. Wir sind im morgendlichen Daily Stand-up. "
            "Verwende ab und zu natürliche österreichische Ausdrücke (wie 'passt schon', 'schauen wir mal', 'eh klar', 'heuer'). "
            "Frage den User, woran er gestern gearbeitet hat, was heute ansteht und ob es Blocker gibt. "
            "WICHTIG: Antworte IMMER kurz (maximal 2-3 gesprochene Sätze), rein als gesprochener Text ohne Sonderzeichen/Markdown."
        ),
    },
    "3": {
        "name": "Alltag & Kaffeeküche (Österreich / Oberösterreich)",
        "prompt": (
            "Du bist ein netter Arbeitskollege aus Linz in der Kaffeepause. Ihr plaudert über Studium, Arbeit, Wochenende oder Alltag in Oberösterreich. "
            "Nutze leichtes österreichisches Deutsch. "
            "WICHTIG: Antworte IMMER kurz (maximal 2-3 gesprochene Sätze), locker und ohne Markdown oder Listen."
        ),
    },
}

FEEDBACK_PROMPT = (
    "Du bist ein präziser Deutschlehrer für B2/C1-Berufssprache in Österreich. "
    "Analysiere ausschließlich die Aussagen des Users aus dem folgenden Gesprächsprotokoll. "
    "Nenne:\n"
    "1. Die 3 wichtigsten Grammatik- oder Ausdrucksfehler (Original -> natürliche B2-Version + kurze Erklärung).\n"
    "2. 2 konkretere Vokabel- oder Phrasen-Upgrades, damit der User im Job noch professioneller klingt.\n"
    "Halte dich kurz, direkt und praxisnah."
)


def record_or_type(whisper: WhisperModel) -> str:
    """Entweder Enter drücken & sprechen, oder direkt Text eintippen."""
    typed = input("\n[ENTER drücken & sprechen, oder Text direkt eintippen]: ").strip()
    if typed:
        return typed

    frames = []
    stop_event = threading.Event()

    def callback(indata, frame_count, time_info, status):
        if not stop_event.is_set():
            frames.append(indata.copy())

    print("🎙️  Aufnahme läuft... (Sprich jetzt & drücke ENTER zum Beenden)")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input()
        stop_event.set()

    if not frames:
        return ""
    audio = np.concatenate(frames, axis=0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)

    segments, _ = whisper.transcribe(buf, language="de", vad_filter=False)
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe(whisper: WhisperModel, audio: np.ndarray) -> str:
    """Transkribiert int16 Numpy-Array über einen In-Memory-WAV-Buffer."""
    if audio.size == 0:
        return ""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)

    segments, _ = whisper.transcribe(buf, language="de", vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


def ask_ollama(messages: list[dict]) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def speak_elevenlabs(client: ElevenLabs | None, text: str):
    if not client:
        return
    try:
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )
        play(audio_stream)
    except Exception as e:
        print(f"[TTS-Hinweis: Konnte Audio nicht abspielen ({e})]")


def main():
    print("=== 🇦🇹 German B2 Voice Coach (Ollama + ElevenLabs) ===")
    print("1: Tech-Interview (Linz AI / Backend)")
    print("2: Daily Stand-up")
    print("3: Alltag & Kaffeeküche")
    choice = input("Modus wählen [1-3, Standard 1]: ").strip() or "1"
    mode = MODES.get(choice, MODES["1"])

    print(f"\nLade Whisper ('{WHISPER_SIZE}')...")
    whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    el_client = None
    if ELEVENLABS_API_KEY and ELEVENLABS_API_KEY != "dein_key_hier":
        el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        print("ElevenLabs TTS: AKTIV (eleven_flash_v2_5)")
    else:
        print("ElevenLabs TTS: INAKTIV (Kein Key in .env gefunden – nur Textausgabe)")

    messages = [{"role": "system", "content": mode["prompt"]}]

    # KI startet das Gespräch
    opening = ask_ollama(messages)
    messages.append({"role": "assistant", "content": opening})
    print(f"\n🤖 Coach: {opening}")
    speak_elevenlabs(el_client, opening)

    print("\n(Tipp: Sag 'Ende' oder 'Feedback', um das Gespräch zu beenden und deine Auswertung zu erhalten.)")

    while True:
        user_text = record_or_type(whisper)

        if not user_text:
            print("⚠️  Nichts verstanden – bitte nochmal versuchen.")
            continue

        print(f"🧑 Du:    {user_text}")
        messages.append({"role": "user", "content": user_text})

        if any(word in user_text.lower() for word in ["ende", "feedback", "stopp", "fertig"]):
            break

        reply = ask_ollama(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"🤖 Coach: {reply}")
        speak_elevenlabs(el_client, reply)

    # --- POST-SESSION B2 FEEDBACK ---
    print("\n" + "=" * 50)
    print("📊 B2-FEEDBACK & KORREKTUREN WERDEN GENERIERT...")
    print("=" * 50)

    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Coach'}: {m['content']}"
        for m in messages
        if m["role"] != "system"
    )
    feedback = ask_ollama([
        {"role": "system", "content": FEEDBACK_PROMPT},
        {"role": "user", "content": transcript},
    ])
    print(feedback)


if __name__ == "__main__":
    main()