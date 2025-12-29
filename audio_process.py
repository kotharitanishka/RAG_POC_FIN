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
    normalised_transcript = preprocess_text(audio_transcripts, "en")
    print("English transcription normalised: ",normalised_transcript)
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
        transcribed_text = response_json.get("transcription", "")
        
        if not transcribed_text:
            print("Warning: Transcription was empty or missing from response.")
        
        print(lang , " transcription : " , transcribed_text)
        normalised_transcript = preprocess_text(transcribed_text, lang)
        print(lang, " transcription normalised: ",normalised_transcript)
        return transcribed_text
            
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        transcribed_text = "" # Set to empty on failure
        
# Ensure the audio file is in a supported format like WAV, MP3, etc.
def load_hindi_audio_and_transcribe(audio_file_path: str, lang : str) : 
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    # Set device (GPU if available, otherwise CPU) and precision
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Specify the pre-trained model ID
    model_id = "Oriserve/Whisper-Hindi2Hinglish-Prime"

    # Load the speech-to-text model with specified configurations
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,        # Use appropriate precision (float16 for GPU, float32 for CPU)
        low_cpu_mem_usage=True,         # Optimize memory usage during loading
        use_safetensors=True            # Use safetensors format for better security
    )
    model.to(device)                    # Move model to specified device

    # Load the processor for audio preprocessing and tokenization
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Create speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={
            "task": "transcribe",       # Set task to transcription
            "language": lang            # Specify English language
        }
    )

    # Process audio file and print transcription
    result = pipe(audio_file_path, return_timestamps=True) 
    audio_transcripts = result["text"]
    print("Hindi transcription: ",audio_transcripts)
    normalised_transcript = preprocess_text(audio_transcripts, "hi")
    print("Hindi transcription normalised: ",normalised_transcript)
    return audio_transcripts 



def preprocess_text(text: str, lang: str) -> str:
    from indicnlp.normalize.indic_normalize import DevanagariNormalizer
    import re
    """Adaptive cleaning for English, Indic, and mixed-language text."""
    
    
    # 2. Indic-specific normalization
    # Apply to Hindi (hi), Marathi (mr), Gujarati (gu), etc.
    if lang.lower() != "en":
        factory = DevanagariNormalizer()
        text = factory.normalize(text)
        
    # 1. Standard cleaning (Remove extra whitespaces/newlines)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Handle English/Hinglish (lowercase English parts for vector consistency)
    # Be careful: Lowercasing Indic script does nothing, so it's safe to run on mixed text
    text = text.lower() 
    
    return text

# print(load_hindi_audio_and_transcribe("resources/Sample Data for AI Testing/46487-Vicky.wav", "hi"))
# print(load_hindi_audio_and_transcribe("resources/Sample Data for AI Testing/69860101-079-Ahmed.wav", "hi"))
# print(load_hindi_audio_and_transcribe("resources/Sample Data for AI Testing/69860429-079Ahmed_1.wav", "hi"))
# print(load_hindi_audio_and_transcribe("resources/Sample Data for AI Testing/69861713-9258-Siddhi.wav", "hi"))
#print(load_indian_audio_and_transcribe("resources/Sample Data for AI Testing/69866593-Jaipur Lokesh.wav", "hi"))
print("hello")