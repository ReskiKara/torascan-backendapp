import os
from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import uvicorn

app = FastAPI()

# ======================
# CONFIGURATION
# ======================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Inisialisasi Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGroq(temperature=0.3, model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# ======================
# RAG ENGINE (LOAD PDF)
# ======================
# Proses ini akan memakan waktu saat startup pertama kali di Vercel
vector_db = None

def init_rag():
    global vector_db
    try:
        # 1. Load PDF
        loader = PyPDFLoader("artefak_toraja.pdf") # Pastikan file ini ada di server
        data = loader.load()
        
        # 2. Chunking (Memecah teks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        # 3. Embedding & Vector Store (FAISS)
        vector_db = FAISS.from_documents(chunks, embeddings)
        print("RAG System Initialized Successfully")
    except Exception as e:
        print(f"Error initializing RAG: {e}")

# Jalankan inisialisasi
init_rag()

# ======================
# ENDPOINTS
# ======================

@app.get("/get-info")
async def get_info(artefak: str, lang: str = "id"):
    """Endpoint untuk deskripsi awal (Initial Scan)"""
    return await process_rag_request(f"Jelaskan secara mendalam tentang artefak {artefak}", lang)

@app.get("/chat")
async def chat(query: str, artefak: str, lang: str = "id"):
    """Endpoint untuk tanya jawab lanjutan"""
    full_query = f"Pertanyaan tentang {artefak}: {query}"
    return await process_rag_request(full_query, lang)

async def process_rag_request(user_query: str, lang: str):
    if not vector_db:
        raise HTTPException(status_code=500, detail="Sistem RAG belum siap.")

    # 1. Similarity Search (Mencari potongan teks paling relevan)
    docs = vector_db.similarity_search(user_query, k=3)
    context = "\n".join([d.page_content for d in docs])

    # 2. Prompt Engineering
    system_msg = "Anda adalah ahli budaya Toraja. Gunakan konteks yang diberikan untuk menjawab."
    if lang == "en":
        prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer in professional English:"
    else:
        prompt = f"Konteks:\n{context}\n\nPertanyaan: {user_query}\nJawab dalam Bahasa Indonesia yang sopan dan informatif:"

    # 3. Generation (Llama 3.1)
    try:
        response = llm.invoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ])
        return {"description": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
