import os
import re

from fastapi import FastAPI, HTTPException

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    temperature=0.3,
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
)


# =========================
# VECTOR DATABASE
# =========================
vector_db = None


# =========================
# NORMALIZATION
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
# HELPER PAGE & PARAGRAPH
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

    if vector_db is None:

        try:
            pdf_path = "artefak_toraja.pdf"

            print("CURRENT DIR:", os.getcwd())
            print("FILES:", os.listdir())
            print("PDF EXISTS:", os.path.exists(pdf_path))

            if not os.path.exists(pdf_path):
                raise Exception("File PDF tidak ditemukan")

            loader = PyPDFLoader(pdf_path)
            data = loader.load()

            print(f"PDF Loaded: {len(data)} pages")

            # =========================
            # GABUNGKAN TEKS + MAPPING HALAMAN
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
            # RECURSIVE CHUNKING
            # =========================
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    ""
                ]
            )

            chunks = []

            artefact_list = sorted(
                VALID_ARTEFACTS,
                key=len,
                reverse=True
            )

            # =========================
            # AMBIL SECTION PER ARTEFAK
            # LALU DIPECAH SECARA RECURSIVE
            # =========================
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

                # Cari batas artefak berikutnya
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

                if len(section) <= 50:
                    continue

                sub_chunks = splitter.split_text(
                    section
                )

                search_start = start_idx

                for idx, sub_chunk in enumerate(sub_chunks):

                    sub_chunk_clean = sub_chunk.strip()

                    if len(sub_chunk_clean) <= 30:
                        continue

                    global_idx = full_text.find(
                        sub_chunk_clean[:80],
                        search_start
                    )

                    if global_idx == -1:
                        global_idx = start_idx

                    page_number = find_page_number(
                        global_idx,
                        page_mapping
                    )

                    paragraph_number = find_paragraph_number(
                        global_idx,
                        page_mapping,
                        full_text
                    )

                    metadata = {
                        "artifact": artefact.lower(),
                        "page": page_number,
                        "paragraph": paragraph_number,
                        "chunk_index": idx + 1
                    }

                    chunks.append(
                        Document(
                            page_content=sub_chunk_clean,
                            metadata=metadata
                        )
                    )

                    search_start = global_idx + len(
                        sub_chunk_clean
                    )

                    print(f"\n===== {artefact.upper()} | CHUNK {idx + 1} =====")
                    print(f"PAGE: {page_number}")
                    print(f"PARAGRAPH: {paragraph_number}")
                    print(sub_chunk_clean[:300])

            print(f"Chunks Created: {len(chunks)}")

            if not chunks:
                raise Exception("Tidak ada chunk yang berhasil dibuat")

            texts = [
                doc.page_content
                for doc in chunks
            ]

            all_embeddings = []

            for text in texts:
                emb = embeddings.embed_query(
                    text
                )

                all_embeddings.append(
                    emb
                )

            print(f"Embeddings Created: {len(all_embeddings)}")

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

            print("RAG Semantic Retrieval Initialized Successfully")

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
    lang: str = "id",
    k: int = 3
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
            "top_k": k,
            "retrieved_contexts": [],
            "retrieved_context": "",
            "page": None,
            "paragraph": None,
            "description": "Artefak tidak ditemukan dalam basis pengetahuan."
        }

    initial_query = (
        f"Jelaskan artefak {clean_name}, "
        f"fungsi, kegunaan, bahan, dan makna budayanya secara ringkas."
    )

    return await process_rag(
        user_query=initial_query,
        artifact_name=clean_name,
        lang=lang,
        top_k=k
    )


# =========================
# CHAT ENDPOINT
# =========================
@app.get("/chat")
async def chat(
    query: str,
    artefak: str,
    lang: str = "id",
    k: int = 3
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
            "top_k": k,
            "retrieved_contexts": [],
            "retrieved_context": "",
            "page": None,
            "paragraph": None,
            "description": "Artefak tidak ditemukan dalam basis pengetahuan."
        }

    return await process_rag(
        user_query=query,
        artifact_name=clean_name,
        lang=lang,
        top_k=k
    )


# =========================
# RAG PROCESS
# =========================
async def process_rag(
    user_query: str,
    artifact_name: str,
    lang: str,
    top_k: int = 3
):

    db = get_vector_db()

    # =========================
    # SEMANTIC RETRIEVAL QUERY
    # =========================
    retrieval_query = f"{artifact_name} {user_query}"

    search_k = max(
        top_k * 4,
        10
    )

    docs_with_scores = db.similarity_search_with_score(
        retrieval_query,
        k=search_k
    )

    # =========================
    # FILTER BERDASARKAN ARTEFAK
    # =========================
    filtered_docs = []

    for doc, score in docs_with_scores:

        metadata_artifact = doc.metadata.get(
            "artifact",
            ""
        ).lower()

        if metadata_artifact == artifact_name.lower():

            filtered_docs.append(
                (doc, score)
            )

        if len(filtered_docs) >= top_k:
            break

    # fallback jika filter kosong
    if not filtered_docs:
        filtered_docs = docs_with_scores[
            :top_k
        ]

    retrieved_contexts = []

    for rank, (doc, score) in enumerate(
        filtered_docs,
        start=1
    ):

        retrieved_contexts.append({
            "rank": rank,
            "context": doc.page_content,
            "page": doc.metadata.get("page"),
            "paragraph": doc.metadata.get("paragraph"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "similarity_score": float(score)
        })

    combined_context = "\n\n".join([
        f"[Context {item['rank']} | "
        f"Halaman {item['page']} | "
        f"Paragraf {item['paragraph']} | "
        f"Chunk {item['chunk_index']}]\n"
        f"{item['context']}"
        for item in retrieved_contexts
    ])

    if not combined_context:
        raise HTTPException(
            status_code=404,
            detail="Context retrieval tidak ditemukan."
        )

    print("\n===== SEMANTIC RETRIEVAL RESULT =====")
    print(f"QUERY: {retrieval_query}")
    print(f"TOP K: {top_k}")

    for item in retrieved_contexts:
        print(
            f"Rank {item['rank']} | "
            f"Page {item['page']} | "
            f"Paragraph {item['paragraph']} | "
            f"Chunk {item['chunk_index']} | "
            f"Score {item['similarity_score']}"
        )
        print(item["context"][:300])

# =========================
# PROMPT
# =========================
    if lang.lower() == "en":

    system_msg = """
You are a Toraja cultural information system.

RULES:
- Answer only based on the retrieved context.
- Use the retrieved context as the main source.
- Do not use information outside the retrieved context.
- Do not hallucinate or invent information.
- If the requested information is not available in the retrieved context,
  answer:
  "The requested information is not available in the retrieved context."
- Re-explain the information naturally and clearly.
- Do not copy the retrieved context word-for-word.
- You may summarize and reorganize the information,
  but the meaning must remain faithful to the retrieved context.
- Do not mix unrelated artifacts.
- Do not add information that contradicts the context.
- Maximum 2 paragraphs.
- Use concise and formal language.
"""

    prompt = f"""
Retrieved Contexts:
{combined_context}

Artifact:
{artifact_name}

User Question:
{user_query}

TASK:
Answer the user question only based on the retrieved contexts above.

RULES:
- Do not use information outside the retrieved context.
- Do not hallucinate or invent information.
- If the information is not found in the retrieved context,
  answer:
  "The requested information is not available in the retrieved context."
- Explain the answer naturally,
  clearly, and informatively.
- Do not copy the context directly.
- You may summarize and reorganize the sentences,
  but the meaning must remain faithful to the retrieved context.
"""

else:

    system_msg = """
Anda adalah sistem informasi artefak budaya Toraja.

ATURAN:
- Jawaban harus hanya berdasarkan konteks retrieval.
- Gunakan informasi pada context retrieval sebagai sumber utama.
- Jangan menggunakan informasi di luar context retrieval.
- Jangan mengarang jawaban.
- Jika informasi yang ditanyakan tidak ditemukan pada context retrieval,
  jawab:
  "Informasi tersebut tidak ditemukan pada konteks retrieval yang tersedia."
- Jelaskan kembali informasi dengan bahasa yang natural dan informatif.
- Jangan menyalin context secara mentah.
- Anda boleh merangkum dan menyusun ulang kalimat,
  tetapi makna harus tetap sesuai dengan context retrieval.
- Jangan mencampur artefak lain yang tidak relevan.
- Jangan membuat informasi yang bertentangan dengan context retrieval.
- Gunakan bahasa Indonesia formal dan mudah dipahami.
- Jawaban maksimal 2 paragraf.
"""

    prompt = f"""
Konteks Retrieval:
{combined_context}

Artefak:
{artifact_name}

Pertanyaan Pengguna:
{user_query}

TUGAS:
Jawab pertanyaan pengguna hanya berdasarkan konteks retrieval di atas.

ATURAN:
- Jangan menggunakan informasi di luar konteks retrieval.
- Jangan mengarang jawaban.
- Jika informasi tidak ditemukan pada konteks retrieval,
  jawab:
  "Informasi tersebut tidak ditemukan pada konteks retrieval yang tersedia."
- Jelaskan jawaban dengan bahasa yang natural,
  informatif, dan mudah dipahami.
- Jangan menyalin context secara mentah.
- Anda boleh merangkum dan menyusun ulang kalimat,
  tetapi makna harus tetap sesuai dengan context retrieval.
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

        first_context = retrieved_contexts[0]

        return {
            "artifact_name": artifact_name,
            "query": user_query,
            "top_k": top_k,

            # Untuk UI lama / context utama
            "retrieved_context": first_context["context"],
            "page": first_context["page"],
            "paragraph": first_context["paragraph"],

            # Untuk UI advanced top-k
            "retrieved_contexts": retrieved_contexts,

            # Jawaban hasil LLM
            "description": response.content
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM ERROR: {str(e)}"
        )
