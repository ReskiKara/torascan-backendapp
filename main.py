import os
from fastapi import FastAPI, HTTPException

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# =========================
# FASTAPI
# =========================
app = FastAPI()

# =========================
# ENVIRONMENT VARIABLES
# =========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY tidak ditemukan")

if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY tidak ditemukan")

# =========================
# EMBEDDING MODEL
# =========================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# =========================
# LLM MODEL
# =========================
llm = ChatGroq(
    temperature=0.3,
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
)

# =========================
# VECTOR DATABASE
# =========================
vector_db = None

def get_vector_db():
    global vector_db

    if vector_db is None:
        try:

            pdf_path = "artefak_toraja.pdf"

            print("CURRENT DIR:", os.getcwd())
            print("FILES:", os.listdir())
            print("PDF EXISTS:", os.path.exists(pdf_path))

            if not os.path.exists(pdf_path):
                raise Exception("File PDF tidak ditemukan")

            # =========================
            # LOAD PDF
            # =========================
            loader = PyPDFLoader(pdf_path)
            data = loader.load()

            print(f"PDF Loaded: {len(data)} pages")

            # =========================
            # CHUNKING
            # =========================
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = text_splitter.split_documents(data)

            print(f"Chunks Created: {len(chunks)}")

            # =========================
            # VECTOR DATABASE
            # =========================
            vector_db = FAISS.from_documents(
                chunks,
                embeddings
            )

            print("RAG System Initialized Successfully")

        except Exception as e:

            print(f"RAG ERROR: {e}")

            raise HTTPException(
                status_code=500,
                detail=f"RAG ERROR: {str(e)}"
            )

    return vector_db

# =========================
# GET INFO ENDPOINT
# =========================
@app.get("/get-info")
async def get_info(artefak: str, lang: str = "id"):

    db = get_vector_db()

    clean_name = artefak.replace("_", " ")

    query = f"Jelaskan secara mendalam tentang artefak {clean_name}"

    return await process_rag(query, clean_name, lang)

# =========================
# CHAT ENDPOINT
# =========================
@app.get("/chat")
async def chat(query: str, artefak: str, lang: str = "id"):

    db = get_vector_db()

    clean_name = artefak.replace("_", " ")

    full_query = f"Pertanyaan user tentang {clean_name}: {query}"

    return await process_rag(full_query, clean_name, lang)

# =========================
# RAG PROCESS
# =========================
async def process_rag(user_query: str, artifact_name: str, lang: str):

    db = get_vector_db()

    # =========================
    # SIMILARITY SEARCH
    # =========================
    docs = db.similarity_search(user_query, k=3)

    # =========================
    # CONTEXT
    # =========================
    context = "\n".join([
        doc.page_content for doc in docs
    ])

    # =========================
    # PROMPT
    # =========================
    if lang.lower() == "en":

        system_msg = (
            "You are a Toraja cultural expert. "
            "Answer only based on the provided context."
        )

        prompt = f"""
Context:
{context}

Artifact:
{artifact_name}

Question:
{user_query}

Answer in professional English.
"""

    else:

        system_msg = (
            "Anda adalah ahli budaya Toraja. "
            "Jawab hanya berdasarkan konteks yang diberikan."
        )

        prompt = f"""
Konteks:
{context}

Artefak:
{artifact_name}

Pertanyaan:
{user_query}

Jawab dalam Bahasa Indonesia formal.
"""

    try:

        response = llm.invoke([
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": prompt
            }
        ])

        return {
            "artifact_name": artifact_name,
            "query": user_query,
            "description": response.content
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"LLM ERROR: {str(e)}"
        )
