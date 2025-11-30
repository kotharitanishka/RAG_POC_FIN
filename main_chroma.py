import chromadb
import os
from dotenv import load_dotenv
from chromadb import Collection

load_dotenv()
chroma_api_key = os.getenv("CHROMA_API_KEY")
chroma_tenant_id = os.getenv("CHROMA_TENANT_ID")
chroma_database = os.getenv("CHROMA_DATABASE_NAME")

def get_chroma_connection_and_collection():
    """Establish and test chroma db connection."""
    client = chromadb.CloudClient(
        api_key=chroma_api_key,
        tenant=chroma_tenant_id,
        database=chroma_database
    )
    collection = None
    try:
        client.heartbeat()
        print("✓ chroma db connected successfully")
        collection = chroma_client.create_collection(name="rag_poc_collection")
        return client, collection
    except Exception as e:
        print(f"✗ chroma db connection failed: {e}")
        raise


def load_data(file_path : str, chunks :list[str]): 
    # Generate required lists
    ids = []
    metadatas = []

    file_name = os.path.basename(file_path)
    safe_prefix = file_name.replace(" ", "_").replace(".", "-").lower()

    for i, chunk in enumerate(chunks):
        # Generate unique ID for the chunk
        unique_id = f"{safe_prefix}_{str(i).zfill(3)}"
        ids.append(unique_id)
        
        # Add metadata to track the source file (CRITICAL for RAG)
        metadatas.append({"source": file_name, "chunk_index": i})
        
    # 3. Add to ChromaDB
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )
    print(f"Successfully added {len(chunks)} chunks from {file_name}.")


def retrive_context(query:str) : 
    # Run the query
    results = collection.query(
        query_texts=[query],
        include=["documents"]
    )

    # Extract the list of document strings
    documents_list = results['documents'][0] 
    
    return documents_list

chroma_client, collection = get_chroma_connection_and_collection()