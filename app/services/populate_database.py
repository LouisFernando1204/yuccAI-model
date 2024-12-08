import os
import shutil
from app.services.get_embedding_function import get_embedding_function
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from colorama import Fore, Style

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")
DATA_PATH = os.path.join(BASE_DIR, "..", "data")  

def populate_database_service(reset: bool = False):
    """
    Populate the Chroma database with embeddings from PDFs.
    If `reset` is True, clear the database before repopulating it.
    """
    if reset:
        clear_database()
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
        )
        print(Fore.YELLOW + "After Clearing:" + Style.RESET_ALL, db.get())
        return "Database reset successfully!"

    documents = load_documents()
    print(Fore.YELLOW + "Documents:" + Style.RESET_ALL, documents)
    chunks = split_documents(documents)
    print(Fore.YELLOW + "Chunks:" + Style.RESET_ALL, chunks)
    add_to_chroma(chunks)
    return "Database populated successfully!"

def load_documents():
    """
    Load PDF documents from the DATA_PATH directory.
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    """
    Split documents into smaller chunks based on specified size and overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    """
    Add document chunks to the Chroma database, avoiding duplicates.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    print(Fore.YELLOW + "Chunks With Ids:" + Style.RESET_ALL, chunks_with_ids)

    existing_items = db.get(include=[]) 
    print(Fore.YELLOW + "Existing Items:" + Style.RESET_ALL, existing_items)
    existing_ids = set(existing_items["ids"])
    print(Fore.YELLOW + "Existing Ids:" + Style.RESET_ALL, existing_ids)
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        print(Fore.YELLOW + "Final DB Now:" + Style.RESET_ALL, db.get())
        # print(Fore.YELLOW + "Embeddings:" + Style.RESET_ALL, db._collection.get(include=['embeddings']))
        
    else:
        print("✅ No new documents to add")

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

def clear_database():
    """
    Clear the Chroma database by deleting the CHROMA_PATH directory.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)