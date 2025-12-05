"""
FastAPI application for RAG (Retrieval-Augmented Generation) system
Provides endpoints to load PDFs and query documents
"""

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
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
    split_text,
    embed_chunks,
    create_async_index,
    load_data_to_index,
    user_query_caching,
    SCHEMA,
    INDEX_NAME
)

# Import audio processing function
from audio_process import load_audio_and_transcribe, load_indian_audio_and_transcribe

app = FastAPI(
    title="RAG POC API",
    description="API for loading PDFs and querying documents using RAG",
    version="1.0.0"
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
    use_cache : bool = True
    use_llm : bool = True


class QueryResponse(BaseModel):
    answer: str
    status: str


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
    else :
        transcribed_text = load_indian_audio_and_transcribe(audio_file_path, lang)
    chunks = split_text(transcribed_text)
    return chunks
    

@app.post("/load-file", response_model=LoadDocumentResponse)
async def load_file(lang: str=Form("en"), file: UploadFile = File(...)):
    """
    Load and process a PDF or audio file.
    
    - **file**: PDF or audio file (supports: .pdf, .mp3, .wav, .m4a, .flac, .ogg) to upload and process
    - Returns: Status message, file type, and number of chunks loaded
    """
    # Supported file extensions
    pdf_extensions = ['.pdf']
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac']
    supported_extensions = pdf_extensions + audio_extensions
    
    # Get file extension
    file_ext = Path(file.filename).suffix.lower() if file.filename else ''
    
    # Validate file type
    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported formats: PDF ({', '.join(pdf_extensions)}) or Audio ({', '.join(audio_extensions)})"
        )
    
    # Determine file type
    is_pdf = file_ext in pdf_extensions
    is_audio = file_ext in audio_extensions
    file_type = "PDF" if is_pdf else "Audio"
    
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
        
        # Create or recreate index
        async_index = create_async_index(SCHEMA)
        keys = load_data_to_index(async_index, chunks, embeddings)
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
    
    - **query**: The question to ask about the documents
    - Returns: Answer based on the document content
    """
    if not _global_state["initialized"]:
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please load a PDF or audio file first using /load-pdf endpoint."
        )
    
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    try:
        # Get answer using RAG
        answer = await answer_question(
            _global_state["async_index"],
            request.query,
            _global_state["cache"], 
            request.use_cache,
            request.use_llm
        )
        
        return QueryResponse(
            answer=answer,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
