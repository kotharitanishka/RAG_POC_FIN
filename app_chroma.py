"""
FastAPI application for RAG (Retrieval-Augmented Generation) system using ChromaDB
Provides endpoints to load PDFs and query documents
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
import shutil
from pathlib import Path
import asyncio

# Import functions from main_chroma.py
import main_chroma
from main_chroma import (
    get_chroma_connection_and_collection,
    load_data,
    retrieve_context
)

# Import functions from main.py (for document loading and LLM)
from main import (
    load_and_split_document,
    generate_llm_response,
    split_text
)

# Import audio processing function
from audio_process import load_audio_and_transcribe

app = FastAPI(
    title="RAG POC API (ChromaDB)",
    description="API for loading PDFs and querying documents using RAG with ChromaDB",
    version="1.0.0"
)

# Global state to store initialized components
_global_state = {
    "chroma_client": None,
    "collection": None,
    "initialized": False
}


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    status: str


class LoadDocumentResponse(BaseModel):
    message: str
    status: str
    file_type: Optional[str] = None
    chunks_loaded: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    """Initialize ChromaDB connection on startup."""
    try:
        chroma_client, collection = get_chroma_connection_and_collection()
        _global_state["chroma_client"] = chroma_client
        _global_state["collection"] = collection
        print("✓ ChromaDB connection verified on startup")
    except Exception as e:
        print(f"⚠ Warning: ChromaDB connection failed on startup: {e}")
        print("⚠ Make sure ChromaDB credentials are configured before loading files")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG POC API (ChromaDB)",
        "endpoints": {
            "load_file": "/load-file (supports PDF and audio files)",
            "query_data": "/query-data",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    chroma_status = "disconnected"
    try:
        if _global_state["chroma_client"]:
            _global_state["chroma_client"].heartbeat()
            chroma_status = "connected"
    except Exception as e:
        chroma_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if chroma_status == "connected" else "degraded",
        "chromadb": chroma_status,
        "rag_initialized": _global_state["initialized"]
    }


def _load_and_split_audio(audio_file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load audio file, transcribe it, and split into chunks."""
    # Transcribe audio to text
    print("Transcribing audio file...")
    transcribed_text = load_audio_and_transcribe(audio_file_path)
    chunks = split_text(transcribed_text)
    return chunks


@app.post("/load-file", response_model=LoadDocumentResponse)
async def load_file(file: UploadFile = File(...)):
    """
    Load and process a PDF or audio file.
    
    - **file**: PDF or audio file (supports: .pdf, .mp3, .wav, .m4a, .flac, .ogg) to upload and process
    - Returns: Status message, file type, and number of chunks loaded
    """
    # Ensure ChromaDB is connected
    if not _global_state["collection"]:
        try:
            chroma_client, collection = get_chroma_connection_and_collection()
            _global_state["chroma_client"] = chroma_client
            _global_state["collection"] = collection
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"ChromaDB connection failed: {str(e)}"
            )
    
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
            chunks = _load_and_split_audio(temp_file_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type"
            )
        
        # Load data to ChromaDB
        load_data(temp_file_path, chunks, _global_state["collection"])
        
        # Update global state
        _global_state["initialized"] = True
        
        print(f"{'='*50}")
        print(f"✓ {file_type} file loaded successfully")
        print(f"{'='*50}\n")
        
        return LoadDocumentResponse(
            message=f"{file_type} file '{file.filename}' loaded successfully",
            status="success",
            file_type=file_type.lower(),
            chunks_loaded=len(chunks)
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
async def query_data(request: QueryRequest):
    """
    Query the loaded documents using RAG.
    
    - **query**: The question to ask about the documents
    - Returns: Answer based on the document content
    """
    if not _global_state["initialized"]:
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please load a PDF or audio file first using /load-file endpoint."
        )
    
    if not _global_state["collection"]:
        raise HTTPException(
            status_code=500,
            detail="ChromaDB collection not available. Please check connection."
        )
    
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    try:
        # Retrieve context from ChromaDB
        context = retrieve_context(request.query, _global_state["collection"])
        
        # Generate LLM response using the retrieved context
        answer = await generate_llm_response(request.query, context)
        
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
    uvicorn.run(app, host="0.0.0.0", port=8001)

