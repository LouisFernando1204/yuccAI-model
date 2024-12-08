from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    """
    Initialize and return the embedding function.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings