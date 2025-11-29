"""
FastAPI application for RAG (Retrieval-Augmented Generation) system
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

# Import functions from main.py
import main
from main import (
    initialize,
    answer_question,
    get_redis_connection,
    create_vectorizer,
    load_and_split_document,
    embed_chunks,
    create_async_index,
    load_data_to_index,
    user_query_caching,
    SCHEMA,
    INDEX_NAME
)

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


class QueryResponse(BaseModel):
    answer: str
    status: str


class LoadPDFResponse(BaseModel):
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
            "load_pdf": "/load-pdf",
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


@app.post("/load-pdf", response_model=LoadPDFResponse)
async def load_pdf(file: UploadFile = File(...)):
    """
    Load and process a PDF file.
    
    - **file**: PDF file to upload and process
    - Returns: Status message and number of chunks loaded
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF (.pdf extension required)"
        )
    
    # Check Redis connection
    try:
        get_redis_connection()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Redis connection failed: {str(e)}. Please ensure Redis is running."
        )
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\n{'='*50}")
        print(f"Processing PDF: {file.filename}")
        print(f"{'='*50}")
        
        # Load and split document
        chunks = load_and_split_document(temp_file_path)
        
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
        
        # Update global state
        _global_state["async_index"] = async_index
        _global_state["initialized"] = True
        
        print(f"{'='*50}")
        print("✓ PDF loaded successfully")
        print(f"{'='*50}\n")
        
        return LoadPDFResponse(
            message=f"PDF '{file.filename}' loaded successfully",
            status="success",
            chunks_loaded=len(keys)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
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
            detail="No PDF loaded. Please load a PDF first using /load-pdf endpoint."
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
            _global_state["cache"]
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
