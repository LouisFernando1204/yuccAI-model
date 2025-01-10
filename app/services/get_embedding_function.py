from langchain_openai import OpenAIEmbeddings

def get_embedding_function():
    """
    Initialize and return the embedding function.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    print("EMBEDDING FUNCTION:", embeddings);
    return embeddings