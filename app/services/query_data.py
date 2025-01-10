import os
from app.services.get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from colorama import Fore, Style

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

PROMPT_TEMPLATE_FIRST = """
Halo! Saya Yucca, asisten digital Universitas Ciputra yang siap membantu Anda. Berikut adalah konteks yang relevan:

{context}

---

Jawablah pertanyaan berikut dengan menggunakan bahasa Indonesia secara alami. Jika memungkinkan, sertakan sapaan ramah atau perkenalan singkat, tetapi tidak harus selalu dimulai dengan frasa tertentu. Pastikan jawaban dimulai dengan perkenalan bahwa Anda adalah Yucca: {question}

Untuk jawaban yang dapat dibuat dalam bentuk poin-poin dan sub poin, maka berikan response atau jawaban dalam bentuk poin-poin dan sub poin apabila perlu.

Jika pertanyaan di luar konteks yang diberikan, tolak dengan sopan dan ramah. Katakan bahwa Anda hanya dapat menjawab pertanyaan sesuai dengan konteks yang diberikan. Berikan opsi kepada pengguna untuk memperjelas pertanyaan atau menambahkan konteks tambahan jika diperlukan.

Sekali lagi, saya adalah Yucca, asisten digital Universitas Ciputra, dan saya siap membantu Anda.
"""

PROMPT_TEMPLATE_FOLLOW_UP = """
Berikut adalah konteks yang relevan:

{context}

---

Jawablah pertanyaan berikut dengan menggunakan bahasa Indonesia secara alami, seperti percakapan sehari-hari, tanpa perkenalan lagi, namun berikan kalimat pembuka untuk mengawali jawabannya: {question}

Untuk jawaban yang dapat dibuat dalam bentuk poin-poin dan sub poin, maka berikan response atau jawaban dalam bentuk poin-poin dan sub poin apabila perlu.

Jika pertanyaan di luar konteks yang diberikan, tolak dengan sopan dan ramah. Katakan bahwa Anda hanya dapat menjawab pertanyaan sesuai dengan konteks yang diberikan. Berikan opsi kepada pengguna untuk memperjelas pertanyaan atau menambahkan konteks tambahan jika diperlukan.

Ingat, saya adalah Yucca, asisten digital Universitas Ciputra.
"""

has_introduced = False
interaction_count = 0  

def query_data_service(query_text: str):
    """
    Query the database and get a response from the model.
    """
    global has_introduced, interaction_count

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    documents_in_db = db.get(include=["documents"]) 
    print(Fore.YELLOW + "Documents in Database:" + Style.RESET_ALL, documents_in_db)

    results = db.similarity_search_with_score(query_text, k=5)
    print(Fore.YELLOW + "Results:" + Style.RESET_ALL, results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(Fore.YELLOW + "Context Text:" + Style.RESET_ALL, context_text)

    if not has_introduced:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_FIRST)
        has_introduced = True  
    else:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_FOLLOW_UP)

    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    response = model([HumanMessage(content=prompt)])
    response_text = response.content

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    pdf_filename = None
    
    if sources:
        first_source = sources[0]  
        pdf_filename = os.path.basename(first_source.split(':')[0])  
    
    formatted_response = f"Response: {response_text}\nPDF File: {pdf_filename}\nSources: {sources}"
    print(Fore.YELLOW + "Model Response:" + Style.RESET_ALL, response_text)
    print(formatted_response)
    
    interaction_count += 1

    if interaction_count >= 10:
        has_introduced = False
        interaction_count = 0 

    return {"answer": response_text, "source": pdf_filename}