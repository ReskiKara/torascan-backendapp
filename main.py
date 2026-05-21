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

@app.on_event("startup")
async def startup_event():
    get_vector_db()

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
# NORMALIZE LABEL
# =========================
def normalize_for_search(text: str) -> str:
    return (
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("'", " ")
        .replace("’", " ")
    )

def normalize_label(text: str) -> str:
    return " ".join(
        normalize_for_search(text).split()
    )

def build_label_pattern(label: str) -> str:
    words = normalize_label(label).split()

    return (
        r"(?<!\w)"
        + r"\s+".join([
            re.escape(word)
            for word in words
        ])
        + r"(?!\w)"
    )

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
    "dolong dolong",
    "pati",
    "bila",
    "gesok gesok",
    "unnuran",
    "tora tora",
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
# HELPER: PAGE & PARAGRAPH
# =========================
def find_page_number(raw_index, page_mapping):
    for mapping in page_mapping:
        if mapping["start"] <= raw_index <= mapping["end"]:
            return mapping["page"]

    return 1

def find_paragraph_number(raw_index, page_mapping, full_text):
    selected_mapping = None

    for mapping in page_mapping:
        if mapping["start"] <= raw_index <= mapping["end"]:
            selected_mapping = mapping
            break

    if not selected_mapping:
        return 1

    page_start = selected_mapping["start"]

    text_before = full_text[
        page_start:raw_index
    ]

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text_before)
        if p.strip()
    ]

    return len(paragraphs) + 1

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
            # GABUNGKAN TEKS PDF + PAGE MAPPING
            # =========================
            full_text = ""
            page_mapping = []

            for i, doc in enumerate(data):

                page_number = i + 1
                page_text = doc.page_content

                start_pos = len(full_text)

                full_text += page_text

                end_pos = len(full_text)

                page_mapping.append({
                    "start": start_pos,
                    "end": end_pos,
                    "page": page_number
                })

                full_text += "\n\n"

            normalized_full_text = normalize_for_search(
                full_text
            )

            # =========================
            # STRUCTURED CHUNKING
            # =========================
            chunks = []

            artefact_list = sorted(
                VALID_ARTEFACTS,
                key=len,
                reverse=True
            )

            for artefact in artefact_list:

                pattern = build_label_pattern(
                    artefact
                )

                match = re.search(
                    pattern,
                    normalized_full_text
                )

                if not match:
                    print(f"TIDAK DITEMUKAN: {artefact}")
                    continue

                start_idx = match.start()

                end_idx = len(full_text)

                # =========================
                # CARI ARTEFAK BERIKUTNYA
                # =========================
                for next_artefact in artefact_list:

                    if next_artefact == artefact:
                        continue

                    next_pattern = build_label_pattern(
                        next_artefact
                    )

                    next_match = re.search(
                        next_pattern,
                        normalized_full_text[
                            start_idx + 1:
                        ]
                    )

                    if next_match:

                        next_idx = (
                            start_idx + 1 +
                            next_match.start()
                        )

                        if next_idx < end_idx:
                            end_idx = next_idx

                section = full_text[
                    start_idx:end_idx
                ].strip()

                if len(section) > 50:

                    page_number = find_page_number(
                        start_idx,
                        page_mapping
                    )

                    paragraph_number = find_paragraph_number(
                        start_idx,
                        page_mapping,
                        full_text
                    )

                    print(f"\n===== {artefact.upper()} =====")
                    print(f"PAGE: {page_number}")
                    print(f"PARAGRAPH: {paragraph_number}")
                    print(section[:500])

                    # =========================
                    # SIMPAN CONTEXT + SOURCE
                    # =========================
                    artifact_chunks[
                        artefact.lower()
                    ] = {
                        "context": section,
                        "page": page_number,
                        "paragraph": paragraph_number
                    }

                    chunks.append(
                        Document(
                            page_content=section,
                            metadata={
                                "artifact": artefact.lower(),
                                "page": page_number,
                                "paragraph": paragraph_number
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

                emb = embeddings.embed_query(
                    text
                )

                all_embeddings.append(
                    emb
                )

            print(f"Embeddings Created: {len(all_embeddings)}")

            # =========================
            # CREATE FAISS DATABASE
            # =========================
            vector_db = FAISS.from_embeddings(
                text_embeddings=list(zip(
                    texts,
                    all_embeddings
                )),
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
async def get_info(
    artefak: str,
    lang: str = "id"
):

    get_vector_db()

    clean_name = normalize_label(
        artefak
    )

    print("LABEL DARI ANDROID:", artefak)
    print("SETELAH NORMALIZE:", clean_name)

    if clean_name not in VALID_ARTEFACTS:

        return {
            "artifact_name": artefak,
            "query": clean_name,
            "retrieved_context": "",
            "page": None,
            "paragraph": None,
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
async def chat(
    query: str,
    artefak: str,
    lang: str = "id"
):

    get_vector_db()

    clean_name = normalize_label(
        artefak
    )

    print("LABEL DARI ANDROID:", artefak)
    print("SETELAH NORMALIZE:", clean_name)

    if clean_name not in VALID_ARTEFACTS:

        return {
            "artifact_name": artefak,
            "query": query,
            "retrieved_context": "",
            "page": None,
            "paragraph": None,
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
async def process_rag(
    user_query: str,
    artifact_name: str,
    lang: str
):

    global artifact_chunks

    # =========================
    # EXACT RETRIEVAL
    # =========================
    retrieved_data = artifact_chunks.get(
        artifact_name.lower()
    )

    if not retrieved_data:

        raise HTTPException(
            status_code=404,
            detail="Context artefak tidak ditemukan."
        )

    context = retrieved_data["context"]
    page_number = retrieved_data["page"]
    paragraph_number = retrieved_data["paragraph"]

    print("\n===== RETRIEVED CONTEXT =====")
    print(f"PAGE: {page_number}")
    print(f"PARAGRAPH: {paragraph_number}")
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
- Jawaban maksimal 2 paragraf.
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

            # =========================
            # CONTEXT RETRIEVAL
            # =========================
            "retrieved_context": context,

            # =========================
            # SOURCE INFO
            # =========================
            "page": page_number,
            "paragraph": paragraph_number,

            # =========================
            # LLM ANSWER
            # =========================
            "description": response.content
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"LLM ERROR: {str(e)}"
        )
