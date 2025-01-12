import os
import time
from app.services.get_embedding_function import get_embedding_function
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
from colorama import Fore, Style
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def init_pinecone():
    """
    Initialize Pinecone and ensure the index exists.
    """
    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index_name = PINECONE_INDEX_NAME

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  
            metric="cosine",  
            spec=ServerlessSpec(
                cloud="aws",  
                region="us-east-1"  
            )
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    return pc, index_name

def populate_database_service(reset: bool = False):
    """
    Populate the Pinecone database with embeddings from PDFs.
    If `reset` is True, clear the database before repopulating it.
    """
    pc, index_name = init_pinecone()  
    index = pc.Index(index_name)
    
    if reset:
        clear_database(pc)  
        print(Fore.YELLOW + "Database cleared!" + Style.RESET_ALL)
        return "Database reset successfully!"

    documents = load_documents()
    print(Fore.YELLOW + "Documents:" + Style.RESET_ALL, documents)
    chunks = split_documents(documents)
    print(Fore.YELLOW + "Chunks:" + Style.RESET_ALL, chunks)
    add_to_pinecone(chunks, index)
    return "Database populated successfully!"

def load_documents():
    """
    Load PDF documents from the DATA_PATH directory.
    """
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    """
    Split documents into smaller chunks based on specified size and overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def add_to_pinecone(chunks: list[Document], index):
    """
    Add or update document chunks in the Pinecone database using upsert.
    """
    embedding_function = get_embedding_function()

    chunks_with_ids = calculate_chunk_ids(chunks)
    print(Fore.YELLOW + "Chunks With Ids:" + Style.RESET_ALL, chunks_with_ids)

    vectors = []
    for chunk in chunks_with_ids:
        metadata = chunk.metadata.copy()
        metadata["page_content"] = chunk.page_content 

        chunk_id = chunk.metadata["id"]
        
        existing_vector = index.fetch(ids=[chunk_id], namespace="yuccai-knowledge")
        if existing_vector.get("vectors"):
            print(f"✅ ID {chunk_id} already exists, skip to another chunk.")
            continue  

        vector = {
            "id": chunk_id,
            "values": embedding_function.embed_query(chunk.page_content),
            "metadata": metadata  
        }
        vectors.append(vector)

    if vectors:
        index.upsert(vectors=vectors, namespace="yuccai-knowledge")
        print(Fore.YELLOW + "Documents upserted to Pinecone!" + Style.RESET_ALL)
    else:
        print("✅ No new documents to upsert")

def calculate_chunk_ids(chunks):
    """
    Generate unique IDs for each chunk based on its source, page, and chunk index.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database(pc):
    """
    Clear the Pinecone database.
    """
    pc.delete_index(PINECONE_INDEX_NAME)