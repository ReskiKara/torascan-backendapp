import os
import re

from fastapi import FastAPI, HTTPException

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# =========================
# FASTAPI
# =========================
app = FastAPI()
@app.get("/")
async def root():
    return {
        "message": "Toraja RAG API Running"
    }

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
# EXACT CONTEXT STORAGE
# =========================
artifact_chunks = {}

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
    global artifact_chunks

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
            # GABUNGKAN TEKS PDF
            # =========================
            full_text = "\n".join([
                doc.page_content
                for doc in data
            ])

            # =========================
            # STRUCTURED CHUNKING
            # BERDASARKAN NAMA ARTEFAK
            # =========================
            chunks = []

            artefact_list = sorted(
                VALID_ARTEFACTS,
                key=len,
                reverse=True
            )

            for artefact in artefact_list:

                match = re.search(
                    rf'\b{re.escape(artefact.lower())}\b',
                    full_text.lower()
                )

                if not match:
                    continue

                start_idx = match.start()

                # Cari artefak berikutnya
                end_idx = len(full_text)

                for next_artefact in artefact_list:

                    if next_artefact == artefact:
                        continue

                    next_match = re.search(
                        rf'\b{re.escape(next_artefact.lower())}\b',
                        full_text.lower()[start_idx + 1:]
                    )

                    if next_match:

                        next_idx = (
                            start_idx + 1 +
                            next_match.start()
                        )

                        if next_idx < end_idx:
                            end_idx = next_idx

                section = full_text[start_idx:end_idx].strip()

                if len(section) > 50:

                    print(f"\n===== {artefact.upper()} =====")
                    print(section[:500])

                    # =========================
                    # SIMPAN EXACT CONTEXT
                    # =========================
                    artifact_chunks[
                        artefact.lower()
                    ] = section

                    chunks.append(
                        Document(
                            page_content=section,
                            metadata={
                                "artifact": artefact.lower()
                            }
                        )
                    )

            print(f"Artefact Chunks Created: {len(chunks)}")

            # =========================
            # TEXT EXTRACTION
            # =========================
            texts = [
                doc.page_content
                for doc in chunks
            ]

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
                embedding=embeddings,
                metadatas=[
                    doc.metadata
                    for doc in chunks
                ]
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

    get_vector_db()

    clean_name = artefak.lower().replace("_", " ").strip()

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

    get_vector_db()

    clean_name = artefak.lower().replace("_", " ").strip()

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

    global artifact_chunks

    # =========================
    # EXACT RETRIEVAL
    # =========================
    context = artifact_chunks.get(
        artifact_name.lower()
    )

    if not context:

        raise HTTPException(
            status_code=404,
            detail="Context artefak tidak ditemukan."
        )

    print("\n===== RETRIEVED CONTEXT =====")
    print(context)

    # =========================
    # PROMPT
    # =========================
    if lang.lower() == "en":

        system_msg = """
You are a Toraja cultural information system.

RULES:
- Answer based on retrieval context.
- Do not mix other artifacts.
- Do not add unrelated information.
- Maximum 2 paragraphs.
- Use concise and formal language.
"""

        prompt = f"""
Context:
{context}

Artifact:
{artifact_name}

User Question:
{user_query}

TASK:
Answer the user question based on the context above.
"""

    else:

        system_msg = """
Anda adalah sistem informasi artefak budaya Toraja.

ATURAN:
- Jawaban harus berdasarkan konteks retrieval.
- Jangan mencampur artefak lain.
- Jangan membuat informasi yang bertentangan dengan konteks.
- Anda boleh memperluas penjelasan secara ringan
  agar lebih informatif dan natural.
- Gunakan bahasa Indonesia formal.
- Jawaban maksimal 2 paragraf.
"""

        prompt = f"""
Konteks:
{context}

Artefak:
{artifact_name}

Pertanyaan Pengguna:
{user_query}

TUGAS:
Jawab pertanyaan pengguna berdasarkan konteks di atas.

ATURAN:
- Jawaban harus sesuai konteks retrieval.
- Jangan mencampur artefak lain.
- Jangan membuat informasi yang bertentangan dengan konteks.
- Anda boleh memperluas penjelasan secara ringan
  agar lebih natural dan informatif.
- Gunakan bahasa Indonesia formal.
- Jawaban maksimal 3 paragraf.
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
