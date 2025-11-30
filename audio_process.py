def _import_audio_transcribe():
    print("Loading audio transcribe  libraries...")
    from transformers import pipeline
    return pipeline

# Load your audio file
# Ensure the audio file is in a supported format like WAV, MP3, etc.
def load_audio_and_transcribe(audio_file_path: str) : 
    pipeline = _import_audio_transcribe()
    # Load the Whisper Small model
    # You can specify a different Whisper model if needed, e.g., "openai/whisper-tiny"
    asr = pipeline("automatic-speech-recognition", "openai/whisper-small")

    # Transcribe the audio
    # The 'chunk_length_s' parameter can be adjusted for longer audio files
    # The 'stride' parameter helps with seamless transcription across chunks
    # The 'return_timestamps' parameter can be set to 'True' to get segment-level timestamps
    result = asr(audio_file_path, chunk_length_s=30, return_timestamps=True)

    # Print the transcribed text
    audio_transcripts = result["text"]
    print(audio_transcripts)
    return audio_transcripts 