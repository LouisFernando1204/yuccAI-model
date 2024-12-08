import os
from app.services.get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from colorama import Fore, Style

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

PROMPT_TEMPLATE = """
Halo! Saya Yucca, asisten digital Universitas Ciputra yang siap membantu Anda dengan informasi yang Anda perlukan. Berikut adalah konteks yang relevan:

{context}

---

Jawablah pertanyaan berikut dengan menggunakan bahasa Indonesia. Pastikan jawaban dimulai dengan sapaan ramah yang memperkenalkan Yucca sebagai bagian dari Universitas Ciputra, misalnya: "Halo, saya Yucca! Senang membantu Anda." Setelah memperkenalkan diri, berikan jawaban berdasarkan konteks yang ada di atas: {question}

Jika pertanyaan di luar konteks yang diberikan, tolak dengan sopan dan ramah. Katakan bahwa Yucca hanya dapat menjawab pertanyaan sesuai dengan konteks yang diberikan. Berikan opsi kepada pengguna untuk memperjelas pertanyaan atau menambahkan konteks tambahan jika diperlukan.
"""

def query_data_service(query_text: str):
    """
    Query the database and get a response from the model.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    documents_in_db = db.get(include=["documents"]) 
    print(Fore.YELLOW + "Documents in Database:" + Style.RESET_ALL, documents_in_db)

    results = db.similarity_search_with_score(query_text, k=5)
    print(Fore.YELLOW + "Results:" + Style.RESET_ALL, results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(Fore.YELLOW + "Context Text:" + Style.RESET_ALL, context_text)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(Fore.YELLOW + "Model Response:" + Style.RESET_ALL, formatted_response)
    return response_text