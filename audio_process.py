import requests


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
    asr = pipeline(
            "automatic-speech-recognition", 
            "openai/whisper-small"
        )

    # Transcribe the audio
    # The 'chunk_length_s' parameter can be adjusted for longer audio files
    # The 'stride' parameter helps with seamless transcription across chunks
    # The 'return_timestamps' parameter can be set to 'True' to get segment-level timestamps
    result = asr(
                audio_file_path, 
                chunk_length_s=30, 
                return_timestamps=True,
                generate_kwargs={"language": "en"}
            )

    # Print the transcribed text
    audio_transcripts = result["text"]
    print("English transcription: ",audio_transcripts)
    return audio_transcripts 

def load_indian_audio_and_transcribe(audio_file_path: str, lang : str) :
    # --- Add POST Request Here ---
    API_URL = 'https://pistillate-birdie-placatingly.ngrok-free.dev/transcribe_audio'
    
    # Prepare the files and data for the POST request
    files = {
        'audio_file': (audio_file_path, open(audio_file_path, 'rb'), 'audio/m4a'),
    }
    data = {
        'locale': lang,
    }
    
    try:
        # Make the POST request
        response = requests.post(API_URL, files=files, data=data)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        # Extract the JSON response
        response_json = response.json()
        
        # Extract the transcription text from the JSON response
        transcribed_text = response_json.get("transcription_ctc", "")
        
        if not transcribed_text:
            print("Warning: Transcription was empty or missing from response.")
        
        print(lang , " transcription : " , transcribed_text)
        return transcribed_text
            
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        transcribed_text = "" # Set to empty on failure

