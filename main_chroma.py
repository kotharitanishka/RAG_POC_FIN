import chromadb
import os
from dotenv import load_dotenv
from chromadb import Collection
import faiss
import datetime
import asyncio


load_dotenv()
chroma_api_key = os.getenv("CHROMA_API_KEY")
chroma_tenant_id = os.getenv("CHROMA_TENANT_ID")
chroma_database = os.getenv("CHROMA_DATABASE_NAME")

from main import (
    load_and_split_document,
    generate_llm_response
)

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
        #use create only first time
        #collection = client.create_collection(name="rag_poc_collection")
        collection = client.get_collection(name="rag_poc_collection")
        
        return client, collection
    except Exception as e:
        print(f"✗ chroma db connection failed: {e}")
        raise


def load_data(file_path : str, chunks :list[str]): 
    # Generate required lists
    ids = []
    doc_content = []
    metadatas = []

    file_name = os.path.basename(file_path)
    safe_prefix = file_name.replace(" ", "_").replace(".", "-").lower()

    for i, chunk in enumerate(chunks):
        # Generate unique ID for the chunk
        unique_id = f"{safe_prefix}_{str(i).zfill(3)}"
        ids.append(unique_id)
        if isinstance(chunk, str):
            doc_content.append(chunk)
        else:
            # Assume it's a Document object with page_content attribute
            doc_content.append(chunk.page_content)
        
        # Add metadata to track the source file (CRITICAL for RAG)
        metadatas.append({"source": file_name, "chunk_index": i})
        
    # 3. Add to ChromaDB
    collection.add(
        ids=ids,
        documents=doc_content,
        metadatas=metadatas
    )
    print(f"Successfully added {len(chunks)} chunks from {file_name}.")


def retrieve_context(query:str) : 
    # Run the query
    results = collection.query(
        query_texts=[query],
        include=["documents"],
        n_results = 5
    )

    # Extract the list of document strings
    documents_list = results['documents'][0] 
    
    formatted_context_list = []
    for result in documents_list:
        formatted_context_list.append(result)

    return "\n\n".join(formatted_context_list)


chroma_client, collection = get_chroma_connection_and_collection()
# filepath = "resources/Questionnaire_for_Improving_Liquidity_in_Bond_Market.pdf"
# chunks = load_and_split_document(filepath)
# load_data(filepath, chunks)
# query = "what are the technologies used"
# results = retrieve_context(query)
# print(datetime.datetime.now())
# llmResults = asyncio.run(generate_llm_response(query, results)) 
# print(datetime.datetime.now())
# print(llmResults)
