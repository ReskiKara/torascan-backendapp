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
    # RETRIEVAL
    # =========================
    docs = db.similarity_search(
        artifact_name,
        k=5
    )

    # =========================
    # FILTER RELEVANT CHUNKS
    # =========================
    relevant_chunks = []

    for doc in docs:

        first_part = doc.page_content.lower()[:150]

        if artifact_name.lower() in first_part:

            relevant_chunks.append(
                doc.page_content
            )

    # =========================
    # FALLBACK
    # =========================
    if not relevant_chunks:

        context = docs[0].page_content

    else:

        context = "\n".join(relevant_chunks)

    print("RETRIEVED CONTEXT:")
    print(context)

    # =========================
    # PROMPT
    # =========================
    if lang.lower() == "en":

        system_msg = f"""
You are a Toraja cultural information system.

FOCUS ONLY ON:
{artifact_name}

RULES:
- Answer ONLY based on retrieval context.
- Do not mix other artifacts.
- Do not add outside information.
- Maximum 2 paragraphs.
- Use concise and formal language.
"""

        prompt = f"""
ANSWER CONTRACT:

1. Artifact Name:
{artifact_name}

2. Use ONLY this context:
{context}

TASK:
Explain the artifact {artifact_name}.

If the context discusses other artifacts,
IGNORE them and only use information
relevant to {artifact_name}.

Answer in concise professional English.
"""

    else:

        system_msg = f"""
Anda adalah sistem informasi artefak budaya Toraja.

FOKUS HANYA PADA:
{artifact_name}

ATURAN:
- Jawab HANYA berdasarkan context retrieval.
- Jangan mencampur artefak lain.
- Jangan menambahkan informasi luar.
- Jawaban maksimal 2 paragraf.
- Gunakan bahasa formal dan ringkas.
"""

        prompt = f"""
KONTRAK JAWABAN:

1. Nama artefak yang harus dijelaskan:
{artifact_name}

2. Gunakan HANYA konteks berikut:
{context}

TUGAS:
Jelaskan artefak {artifact_name}.

Jika konteks membahas artefak lain,
ABAIKAN dan gunakan hanya informasi
yang relevan dengan {artifact_name}.

Jawaban maksimal 2 paragraf.
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
