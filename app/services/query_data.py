import os
import time
from app.services.get_embedding_function import get_embedding_function
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from colorama import Fore, Style
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

has_introduced = False
interaction_count = 0

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

def query_data_service(query_text: str):
    """
    Query the Pinecone database and get a response from the model.
    """
    global has_introduced, interaction_count

    pc, index_name = init_pinecone()  
    index = pc.Index(index_name)

    embedding_function = get_embedding_function()

    embedding = embedding_function.embed_query(query_text)
    
    results = index.query(
        namespace="yuccai-knowledge",  
        vector=embedding,  
        top_k=5,  
        include_values=False, 
        include_metadata=True 
    )

    print(Fore.YELLOW + "Results:" + Style.RESET_ALL, results)

    context_text = "\n\n---\n\n".join([doc['metadata'].get('page_content', '') for doc in results['matches']])
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

    sources = [doc['metadata'].get("id", None) for doc in results['matches']]
    print(Fore.YELLOW + "Sources:" + Style.RESET_ALL, sources)

    selected_sources = [doc['metadata'].get("id") for doc in sorted(results['matches'], key=lambda x: x['score'], reverse=True)]

    pdf_filename = None
    if selected_sources:
        first_source = selected_sources[0]
        pdf_filename = os.path.basename(first_source.split(':')[0])

    formatted_response = f"Response: {response_text}\nPDF File: {pdf_filename}\nSources: {selected_sources}"
    print(Fore.YELLOW + "Model Response:" + Style.RESET_ALL, response_text)
    print(formatted_response)

    interaction_count += 1

    if interaction_count >= 10:
        has_introduced = False
        interaction_count = 0

    return {"answer": response_text, "source": pdf_filename, "sources": selected_sources}