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
    model="gemini-embedding-2",
    google_api_key=GOOGLE_API_KEY
)

# =========================
# LLM MODEL
# =========================
llm = ChatGroq(
    temperature=0.2,
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
)

# =========================
# VECTOR DATABASE
# =========================
vector_db = None

# =========================
# VALID ARTEFACT LABELS
# =========================
VALID_ARTEFACTS = {
    "kandean dulang",
    "ullin",
    "kurin",
    "kalobe",
    "irusan kararo",
    "irusan suke",
    "toka",
    "dolong-dolong",
    "pa'ti",
    "bila",
    "gesok-gesok",
    "unnuran",
    "tora-tora",
    "kandian",
    "dakdak",
    "baka",
    "bakabua",
    "timbo",
    "rakke",
    "gandang",
    "sarong",
    "bombongan",
    "ponto balusu"
}

# =========================
# LOAD VECTOR DATABASE
# =========================
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
                chunk_size=300,
                chunk_overlap=50
            )

            chunks = text_splitter.split_documents(data)

            print(f"Chunks Created: {len(chunks)}")

            # =========================
            # TEXT EXTRACTION
            # =========================
            texts = [doc.page_content for doc in chunks]

            # =========================
            # MANUAL EMBEDDING
            # =========================
            all_embeddings = []

            for text in texts:

                emb = embeddings.embed_query(text)

                all_embeddings.append(emb)

            print(f"Embeddings Created: {len(all_embeddings)}")

            # =========================
            # CREATE FAISS DATABASE
            # =========================
            vector_db = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, all_embeddings)),
                embedding=embeddings
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

    clean_name = artefak.lower().replace("_", " ").strip()

    # =========================
    # VALIDASI ARTEFAK
    # =========================
    if clean_name not in VALID_ARTEFACTS:

        return {
            "artifact_name": artefak,
            "description": "Artefak tidak ditemukan dalam basis pengetahuan."
        }

    return await process_rag(
        user_query=clean_name,
        artifact_name=clean_name,
        lang=lang
    )

# =========================
# CHAT ENDPOINT
# =========================
@app.get("/chat")
async def chat(query: str, artefak: str, lang: str = "id"):

    db = get_vector_db()

    clean_name = artefak.lower().replace("_", " ").strip()

    # =========================
    # VALIDASI ARTEFAK
    # =========================
    if clean_name not in VALID_ARTEFACTS:

        return {
            "artifact_name": artefak,
            "description": "Artefak tidak ditemukan dalam basis pengetahuan."
        }

    return await process_rag(
        user_query=query,
        artifact_name=clean_name,
        lang=lang
    )

# =========================
# RAG PROCESS
# =========================
async def process_rag(user_query: str, artifact_name: str, lang: str):

    db = get_vector_db()

    # =========================
    # RETRIEVAL BERDASARKAN
    # NAMA ARTEFAK
    # =========================
    docs = db.similarity_search(
        artifact_name,
        k=1
    )

    # =========================
    # CONTEXT
    # =========================
    context = "\n".join([
        doc.page_content for doc in docs
    ])

    print("RETRIEVED CONTEXT:")
    print(context)

    # =========================
    # PROMPT
    # =========================
    if lang.lower() == "en":

        system_msg = """
You are a Toraja cultural information system.

RULES:
- Answer only based on retrieval context.
- Do not add outside information.
- Do not mix other artifacts.
- If information is unavailable,
  answer:
  'Information not found in knowledge base.'
- Maximum 2 paragraphs.
- Use concise and formal language.
"""

        prompt = f"""
Context:
{context}

Artifact:
{artifact_name}

Question:
{user_query}

Give a concise explanation in maximum 2 paragraphs.
"""

    else:

        system_msg = """
Anda adalah sistem informasi artefak budaya Toraja.

ATURAN:
- Jawab hanya berdasarkan konteks retrieval.
- Jangan menambahkan informasi di luar konteks.
- Jangan mencampur artefak lain.
- Jika informasi tidak ditemukan dalam konteks,
  jawab:
  'Informasi tidak ditemukan dalam basis pengetahuan.'
- Jawaban maksimal 2 paragraf.
- Gunakan bahasa formal dan ringkas.
"""

        prompt = f"""
Konteks:
{context}

Artefak:
{artifact_name}

Pertanyaan:
{user_query}

Berikan penjelasan singkat maksimal 2 paragraf.
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
