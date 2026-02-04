"""
FastAPI application for RAG (Retrieval-Augmented Generation) system
Provides endpoints to load PDFs and query documents
"""

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
import shutil
from pathlib import Path

# Import functions from main.py
import main
from main import (
    initialize,
    answer_question,
    get_redis_connection,
    create_vectorizer,
    load_and_split_document,
    load_and_split_docx,
    load_and_split_docx_new,
    load_and_split_txt,
    load_data_to_index_new,
    split_text,
    embed_chunks,
    create_async_index,
    load_data_to_index,
    user_query_caching,
    SCHEMA,
    INDEX_NAME
)

# Import audio processing function
from audio_process import load_audio_and_transcribe, load_indian_audio_and_transcribe, load_hindi_audio_and_transcribe

app = FastAPI(
    title="RAG POC API",
    description="API for loading PDFs and querying documents using RAG",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],  # Allows all headers
)

# Global state to store initialized components
_global_state = {
    "async_index": None,
    "vectorizer": None,
    "cache": None,
    "initialized": False
}


class QueryRequest(BaseModel):
    query: str
    use_cache: bool = False
    use_llm: bool = False
    original_query: str = None
    source_lang: str = "en"
    translate_rag: bool = False  # NEW: Option to translate RAG response


class QueryResponse(BaseModel):
    answer: str
    status: str
    original_rag_context: str = None  # NEW: Original English RAG context
    translated_rag_context: str = None  # NEW: Translated RAG context


class LoadDocumentResponse(BaseModel):
    message: str
    status: str
    chunks_loaded: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup."""
    try:
        get_redis_connection()
        print("✓ Redis connection verified on startup")
    except Exception as e:
        print(f"⚠ Warning: Redis connection failed on startup: {e}")
        print("⚠ Make sure Redis is running before loading PDFs")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG POC API",
        "endpoints": {
            "load_document": "/load-pdf (supports PDF and audio files)",
            "query": "/query",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_status = "disconnected"
    try:
        get_redis_connection()
        redis_status = "connected"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if redis_status == "connected" else "degraded",
        "redis": redis_status,
        "rag_initialized": _global_state["initialized"]
    }


def _load_and_split_audio(audio_file_path: str, lang:str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load audio file, transcribe it, and split into chunks."""
    
    # Transcribe audio to text
    print("Transcribing audio file...")
    if lang.lower() == "en":
            print("This is the English transcription")
    transcribed_text = ""
    if lang=="en" :
        transcribed_text = load_audio_and_transcribe(audio_file_path)
    elif lang == "hi" : 
        transcribed_text = load_hindi_audio_and_transcribe(audio_file_path, lang)
    else :
        transcribed_text = load_indian_audio_and_transcribe(audio_file_path, lang)
    chunks = split_text(transcribed_text)
    return chunks
    

@app.post("/load-file", response_model=LoadDocumentResponse)
async def load_file(lang: str=Form("en"), file: UploadFile = File(...)):
    """
    Load and process a PDF, Word, TXT, or audio file.
    
    - **file**: Document or audio file to upload and process
      - Documents: .pdf, .docx, .doc, .txt
      - Audio: .mp3, .wav, .m4a, .flac, .ogg, .wma, .aac
    - Returns: Status message, file type, and number of chunks loaded
    """
    # Supported file extensions
    pdf_extensions = ['.pdf']
    doc_extensions = ['.docx', '.doc']
    txt_extensions = ['.txt']
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac']
    document_extensions = pdf_extensions + doc_extensions + txt_extensions
    supported_extensions = document_extensions  + audio_extensions
    
    # Get file extension
    file_ext = Path(file.filename).suffix.lower() if file.filename else ''
    
    # Validate file type
    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported formats: PDF ({', '.join(pdf_extensions)}), Word ({', '.join(doc_extensions)}), Text ({', '.join(txt_extensions)}), or Audio ({', '.join(audio_extensions)})"
        )
    
    # Determine file type
    is_pdf = file_ext in pdf_extensions
    is_docx = file_ext in doc_extensions
    is_txt = file_ext in txt_extensions
    is_audio = file_ext in audio_extensions
    
    if is_pdf:
        file_type = "PDF"
    elif is_docx:
        file_type = "Word"
    elif is_txt:
        file_type = "Text"
    elif is_audio:
        file_type = "Audio"
    else:
        file_type = "Unknown"
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\n{'='*50}")
        print(f"Processing {file_type} file: {file.filename}")
        print(f"{'='*50}")
        
        # Load and split document based on file type
        if is_pdf:
            chunks = load_and_split_document(temp_file_path)
        elif is_docx:
            #chunks = load_and_split_docx(temp_file_path)
            chunks = load_and_split_docx_new(temp_file_path)
        elif is_txt:
            chunks = load_and_split_txt(temp_file_path)
        elif is_audio:
            chunks = _load_and_split_audio(temp_file_path, lang)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type"
            )
        
        # Create vectorizer if not exists
        if _global_state["vectorizer"] is None:
            _global_state["vectorizer"] = create_vectorizer()
            # Set global vectorizer in main.py for embed_query function
            main._vectorizer = _global_state["vectorizer"]
            _global_state["cache"] = user_query_caching(_global_state["vectorizer"])
        
        # Embed chunks
        embeddings = embed_chunks(_global_state["vectorizer"], chunks)
        
        # --- CACHE CLEARING LOGIC ---
        # Extract cache objects from global state
        embed_cache_to_clear = _global_state["vectorizer"].cache if _global_state["vectorizer"] else None
        llm_cache_to_clear = _global_state["cache"]
        
        # Create or recreate index
        # async_index = create_async_index(SCHEMA)
        # Pass these to the updated create_async_index function
        async_index = create_async_index(
            SCHEMA, 
            embed_cache=embed_cache_to_clear, 
            llm_cache=llm_cache_to_clear
        )
        #keys = load_data_to_index(async_index, chunks, embeddings)
        keys = load_data_to_index_new(async_index, chunks, embeddings)
        #load_data_to_chroma(chunks,filepath)
        
        # Update global state
        _global_state["async_index"] = async_index
        _global_state["initialized"] = True
        
        print(f"{'='*50}")
        print(f"✓ {file_type} file loaded successfully")
        print(f"{'='*50}\n")
        
        return LoadDocumentResponse(
            message=f"{file_type} file '{file.filename}' loaded successfully",
            status="success",
            file_type=file_type.lower(),
            chunks_loaded=len(keys)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing {file_type} file: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the loaded documents using RAG.
    
    - **query**: The question to ask about the documents (English)
    - **original_query**: Original query in user's language (optional)
    - **source_lang**: Source language code (default: "en")
    - **translate_rag**: Return both original and translated RAG context
    - Returns: Answer based on the document content
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
        
    # Create vectorizer if not exists
    async_index = create_async_index(SCHEMA)
    if _global_state["vectorizer"] is None:
        _global_state["vectorizer"] = create_vectorizer()
        main._vectorizer = _global_state["vectorizer"]
        _global_state["cache"] = user_query_caching(_global_state["vectorizer"])
    _global_state["async_index"] = async_index
    
    try:
        if request.source_lang != "en":
            #english_text = translate_py(original_text, source_lang, "en")
            request.query = translate_google(request.original_query, request.source_lang, "en")
        # Get answer using RAG
        result = await answer_question(
            _global_state["async_index"],
            request.query,
            _global_state["cache"], 
            request.use_cache,
            request.use_llm,
            original_query=request.original_query,
            source_lang=request.source_lang,
            translate_rag=request.translate_rag  # NEW
        )
        
        # Handle different return types
        if isinstance(result, dict):
            return QueryResponse(
                answer=result['answer'],
                status="success",
                original_rag_context=result.get('original_rag_context'),
                translated_rag_context=result.get('translated_rag_context')
            )
        else:
            return QueryResponse(
                answer=result,
                status="success"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/audio-to-text")
async def audio_to_text(
    audio: UploadFile = File(...),
    source_lang: str = "en"  # Default to English, can be 'hi', 'ta', 'te', etc.
):
    """
    Convert audio file to text using Google Speech Recognition.
    
    Args:
        audio: Audio file (WAV, WebM, MP3, etc.)
        source_lang: Language code (en, hi, ta, te, bn, mr, gu, kn, ml)
    
    Returns:
        JSON with transcribed text and metadata
    
    Example:
        {
            "text": "What is BSDA?",
            "language": "en",
            "eglish_text: "What is BSDA?",
            "confidence": 0.95,
            "status": "success"
        }
    """
    import speech_recognition as sr
    from pydub import AudioSegment
    from translatepy import Translator
    print(f"\n{'='*60}")
    print(f"🎤 Audio-to-Text Request")
    print(f"{'='*60}")
    print(f"File: {audio.filename}")
    print(f"Content-Type: {audio.content_type}")
    print(f"Language: {source_lang}")
    
    try:
        # Read audio file
        audio_data = await audio.read()
        print(f"Audio size: {len(audio_data)} bytes")
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
         # Initialize translator
        translator = Translator()
        
        # Optimize recognizer settings
        recognizer.energy_threshold = 4000
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        
        # Create temporary files
        input_ext = os.path.splitext(audio.filename)[1] or '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_ext) as temp_input:
            temp_input.write(audio_data)
            temp_input_path = temp_input.name
        
        wav_path = None
        
        try:
            # Convert to WAV if needed (WebM, MP3, etc.)
            if input_ext.lower() not in ['.wav']:
                print(f"🔄 Converting {input_ext} to WAV...")
                audio_segment = AudioSegment.from_file(temp_input_path)
                
                # Export as WAV
                wav_path = temp_input_path.replace(input_ext, '.wav')
                audio_segment.export(
                    wav_path,
                    format='wav',
                    parameters=["-ar", "16000", "-ac", "1"]  # 16kHz, mono
                )
                file_to_process = wav_path
                print(f"✅ Converted to WAV: {wav_path}")
            else:
                file_to_process = temp_input_path
            
            # Convert audio to AudioData
            with sr.AudioFile(file_to_process) as source:
                print("📝 Processing audio...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_content = recognizer.record(source)
            
            # Speech to text using Google
            print(f"🔄 Transcribing in {source_lang}...")
            original_text  = recognizer.recognize_google(
                audio_content, 
                language=source_lang,
                show_all=False  # Set True to get confidence scores
            )
            
            print(f"✅ Transcribed: {original_text }")
            
             # Translate to English if needed
            if source_lang != "en":
                #english_text = translate_py(original_text, source_lang, "en")
                english_text = translate_google(original_text, source_lang, "en")
            else:
                english_text = original_text
        
            
            return {
                "text": original_text,
                "language": source_lang,
                "english_text" : english_text,
                "status": "success",
                "message": "Audio transcribed successfully"
            }
        
        finally:
            # Cleanup temp files
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
    
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return {
            "text": "",
            "error": "Could not understand audio - please speak clearly",
            "status": "error"
        }
    
    except sr.RequestError as e:
        print(f"❌ Speech recognition service error: {e}")
        return {
            "text": "",
            "error": f"Speech recognition service error: {str(e)}",
            "status": "error"
        }
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {
            "text": "",
            "error": f"Processing error: {str(e)}",
            "status": "error"
        }
        
def translate_py(text : str , source_lang : str = "en" , target_lang : str = "en") :
    from translatepy import Translator
    try : 
        # Initialize translator
        translator = Translator()
        print(f"🔄 Translating from {source_lang} to {target_lang}")
        result = translator.translate(text, target_lang, source_lang)
        final_text = result.result
        print(f"✅ Translated: {final_text}")
        return final_text
    except Exception as trans_error:
        print(f"⚠️ Translation failed, using original text: {trans_error}")
        final_text = text
        return final_text
    
from deep_translator import GoogleTranslator

# Initialize outside to reuse the connection session
def translate_google(text: str, source_lang: str = "en", target_lang: str = "en"):
    print(f"🔄 Translating {text[:20]}... from {source_lang} to {target_lang}")
    try:
        # 'auto' is better for the source if you aren't 100% sure it's English
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        final_text = translator.translate(text)
        
        if not final_text:
            return text
            
        print(f"✅ Translated: {final_text}")
        return final_text
    except Exception as trans_error:
        print(f"⚠️ Translation failed: {trans_error}")
        return text


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6901)
