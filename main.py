"""
RAG (Retrieval-Augmented Generation) Proof of Concept
Uses Redis Vector Search with LangChain and Anthropic Claude

Optimized with lazy imports for faster startup.
"""

import os
import asyncio
import warnings
import getpass
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

# Only lightweight imports at startup
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# Configuration
# =============================================================================

REDIS_HOST = "localhost"
REDIS_PORT = "6379"
REDIS_PASSWORD = ""
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

CHAT_MODEL = "claude-sonnet-4-5-20250929"
INDEX_NAME = "redisvl"
DOC_PATH = "resources/Questionnaire_for_Improving_Liquidity_in_Bond_Market.pdf"

SYSTEM_PROMPT = """
You are an AI assistant that answers questions using ONLY the retrieved context 
from the user's PDFs. If an answer is not fully supported by the context, 
you must respond: "I could not find this information in the provided documents."

Rules:
- Never make up information.
- Never use external knowledge unless the user asks for it explicitly.
- Prefer quoting or summarizing retrieved passages when possible.
- Stay factual, concise, and grounded in the provided text.
- If context is unclear or conflicting, state the ambiguity.
"""

# Index schema for Redis Vector Search
SCHEMA = {
    "index": {
        "name": INDEX_NAME,
        "prefix": "chunk",
        "storage_type": "hash"
    },
    "fields": [
        {"name": "chunk_id", "type": "tag", "attrs": {"sortable": True}},
        {"name": "content", "type": "text"},
        {
            "name": "text_embedding",
            "type": "vector",
            "attrs": {
                "dims": 384,
                "distance_metric": "cosine",
                "algorithm": "hnsw",
                "datatype": "float32"
            }
        }
    ]
}

# Global references (set during initialization)
_vectorizer = None
_async_index = None


# =============================================================================
# Lazy Import Helpers
# =============================================================================

def _import_redis():
    from redis import Redis
    return Redis


def _import_pdf_loader():
    print("Loading PDF processing libraries...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    return RecursiveCharacterTextSplitter, PyPDFLoader

def _import_split_text():
    print("Loading text splitting processing libraries...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter


def _import_vectorizer():
    print("Loading embedding model (this may take a minute on first run)...")
    from redisvl.utils.vectorize import HFTextVectorizer
    from redisvl.extensions.cache.embeddings import EmbeddingsCache
    return HFTextVectorizer, EmbeddingsCache


def _import_index():
    from redisvl.index import SearchIndex, AsyncSearchIndex
    from redisvl.query import VectorQuery
    from redisvl.redis.utils import array_to_buffer
    return SearchIndex, AsyncSearchIndex, VectorQuery, array_to_buffer


def _import_anthropic():
    import anthropic
    return anthropic


# =============================================================================
# Redis Connection
# =============================================================================

def get_redis_connection():
    """Establish and test Redis connection."""
    Redis = _import_redis()
    redis_connection = Redis.from_url(
        REDIS_URL,
        socket_timeout=2,
        socket_connect_timeout=2,
    )
    try:
        redis_connection.ping()
        print("✓ Redis connected successfully")
        return redis_connection
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        raise


# =============================================================================
# Document Processing
# =============================================================================

def load_and_split_document(doc_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load PDF and split into chunks."""
    RecursiveCharacterTextSplitter, PyPDFLoader = _import_pdf_loader()
    
    loader = PyPDFLoader(doc_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = loader.load_and_split(text_splitter)
    print(f"✓ Created {len(chunks)} chunks from: {doc_path}")
    return chunks


# =============================================================================
# Chunking plain text 
# =============================================================================

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """split text into chunks."""
    RecursiveCharacterTextSplitter = _import_split_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    print(f"✓ Created {len(chunks)} chunks from the text provided")
    return chunks


# =============================================================================
# Embedding & Vectorization
# =============================================================================

def create_vectorizer():
    """Create HuggingFace text vectorizer with embedding cache."""
    HFTextVectorizer, EmbeddingsCache = _import_vectorizer()
    
    hf = HFTextVectorizer(
        model="sentence-transformers/all-MiniLM-L6-v2",
        cache=EmbeddingsCache(
            name="embedcache",
            ttl=600,
            redis_url=REDIS_URL,
        )
    )
    print("✓ Vectorizer ready")
    return hf


def embed_chunks(vectorizer, chunks) -> list:
    """Embed all document chunks."""
    print("Embedding chunks...")
    # Handle both string chunks and Document objects
    texts = []
    for chunk in chunks:
        if isinstance(chunk, str):
            texts.append(chunk)
        else:
            # Assume it's a Document object with page_content attribute
            texts.append(chunk.page_content)
    embeddings = vectorizer.embed_many(texts)
    assert len(embeddings) == len(chunks), "Embedding count mismatch"
    print(f"✓ Embedded {len(embeddings)} chunks")
    return embeddings


# =============================================================================
# Index Management
# =============================================================================

def create_async_index(schema: dict):
    """Create Redis search index from schema."""
    AsyncSearchIndex, _, _, _ = _import_index()
    
    index = AsyncSearchIndex.from_dict(schema, redis_url=REDIS_URL)
    index.create(overwrite=True, drop=True)
    print(f"✓ Async Index '{schema['index']['name']}' created")
    return index


def load_data_to_index(index, chunks, embeddings) -> list:
    """Load embedded chunks into the Redis index."""
    _, _, _, array_to_buffer = _import_index()
    
    data = []
    for i, chunk in enumerate(chunks):
        # Handle both string chunks and Document objects
        if isinstance(chunk, str):
            content = chunk
        else:
            # Assume it's a Document object with page_content attribute
            content = chunk.page_content
        
        data.append({
            'chunk_id': i,
            'content': content,
            'text_embedding': array_to_buffer(embeddings[i], dtype='float32')
        })
    
    keys = index.load(data, id_field="chunk_id")
    print(f"✓ Loaded {len(keys)} chunks into index")
    return keys


# =============================================================================
# Query Functions
# =============================================================================

def embed_query(query: str) -> list:
    """Convert a user query into a dense vector representation."""
    return _vectorizer.embed(query)

def user_query_caching(hf):
    from redisvl.extensions.cache.llm import SemanticCache

    llmcache = SemanticCache(
        name="llmcache",                
        vectorizer=hf,                  
        redis_url=REDIS_URL,          
        ttl=600,                         
        distance_threshold=0.45,        
        overwrite=True      
    )

    print("Cache created successfully:", llmcache)
    return llmcache

async def retrieve_context(async_index, query_vector) -> str:
    """Fetch the relevant context from Redis using vector search."""
    _, _, VectorQuery, _ = _import_index()
    
    results = async_index.query(
        VectorQuery(
            vector=query_vector,
            vector_field_name="text_embedding",
            return_fields=["content"],
            num_results=5
        )
    )

    formatted_context_list = []
    for result in results:
        content = result.get("content", "")
        formatted_context_list.append(content)

    return "\n\n".join(formatted_context_list)


# =============================================================================
# LLM Response Generation
# =============================================================================

def promptify(query: str, context: str, domain: str = "general") -> str:
    """Generates a prompt for RAG."""
    return f'''You are an expert {domain} assistant. 
    Use the provided context below to answer the user's question.
    
    - Base your answer ONLY on the provided context.
    - If you cannot answer based on the context, do not guess.
    
    User question:
    {query}

    Helpful context:
    {context}

    Answer:
    '''


async def generate_llm_response(query: str, context: str) -> str:
    """Send prompt to Anthropic LLM and return response."""
    anthropic = _import_anthropic()
    
    client = anthropic.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model=CHAT_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": promptify(query, context)}],
        temperature=0.1,
    )
    return response.content[0].text


async def answer_question(index, query: str , cache) -> str:
    """End-to-end RAG: embeds query, retrieves context, generates LLM response."""
    query_vector = embed_query(query)
    
    results = cache.check(vector=query_vector)
    
    if results:
        print("found similar, semantic")
        return results[0]['response']

    context = await retrieve_context(index, query_vector)

    
    llmResults = await generate_llm_response(query, context)
    cache.store(query, llmResults, query_vector)
    return llmResults


# =============================================================================
# Initialization
# =============================================================================

def initialize(doc_path: str = DOC_PATH):
    """Initialize the RAG system."""
    global _vectorizer, _async_index
    _, AsyncSearchIndex, _, _ = _import_index()

    print("\n" + "=" * 50)
    print("Initializing RAG System")
    print("=" * 50)

    # 1. Test Redis
    get_redis_connection()

    # 2. Load document
    chunks = load_and_split_document(doc_path)

    # 3. Create vectorizer
    _vectorizer = create_vectorizer()
    
    llmCache = user_query_caching(_vectorizer)

    # 4. Embed chunks
    embeddings = embed_chunks(_vectorizer, chunks)

    # 5. Create & load index
    async_index = create_async_index(SCHEMA)
    load_data_to_index(async_index, chunks, embeddings)

    print("=" * 50)
    print("✓ RAG System Ready!")
    print("=" * 50 + "\n")

    return async_index,llmCache


def ensure_api_key():
    """Ensure Anthropic API key is set."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("ANTHROPIC_API_KEY: ")
    print(f"Using model: {CHAT_MODEL}")


async def run_test_questions(async_index, questions: list , cache):
    """Run test questions through the RAG system."""
    results = await asyncio.gather(*[
        answer_question(async_index, q , cache) for q in questions
    ])

    for i, result in enumerate(results):
        print(f"\nQ: {questions[i]}")
        print(f"A: {result}")
        print("-" * 50)


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    print("RAG PoC Starting...")
    
    # Ensure API key
    #ensure_api_key()

    # Initialize (lazy loading happens here)
    async_index,llmCache = initialize()
    
    # Test
    questions = ["In this project, list technologies utilized"]
    asyncio.run(run_test_questions(async_index, questions , llmCache))




if __name__ == "__main__":
    main()