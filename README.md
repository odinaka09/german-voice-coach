# 🇦🇹 German B2 Voice Coach

A low-latency CLI voice tutor built to practice spoken German (B2 level) and Austrian workplace communication for software and AI engineering roles.

## Architecture
- **Speech-to-Text (STT):** Local `faster-whisper` (`small`, INT8) with push-to-talk recording via `sounddevice`
- **Conversation Engine (LLM):** Local Ollama (`gemma3:4b`) with scenario-specific system prompts (Tech Interview, Daily Stand-up, Austrian Office Smalltalk)
- **Text-to-Speech (TTS):** ElevenLabs API (`eleven_flash_v2_5`) with graceful text-only fallback
- **Post-Session Analysis:** Automated B2 grammar & vocabulary feedback generated from the session transcript

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
