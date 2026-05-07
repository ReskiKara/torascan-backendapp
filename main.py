import os
from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

app = FastAPI()

# Ambil API Key dari Environment Variable Vercel
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Inisialisasi Model secara global (API Based)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=GOOGLE_API_KEY
)
llm = ChatGroq(
    temperature=0.3, 
    model_name="llama-3.1-8b-instant", 
    groq_api_key=GROQ_API_KEY
)

# Variable global untuk menyimpan Vector Database
vector_db = None

def get_vector_db():
    """Fungsi untuk inisialisasi RAG (Chunking + Embedding)"""
    global vector_db
    if vector_db is None:
        try:
            # Cari path file PDF
            pdf_path = os.path.join(os.path.dirname(__file__), "artefak_toraja.pdf")
            if not os.path.exists(pdf_path):
                print(f"Error: File {pdf_path} tidak ditemukan!")
                return None
            
            # 1. LOAD PDF
            loader = PyPDFLoader(pdf_path)
            data = loader.load()
            
            # 2. CHUNKING (Memecah dokumen agar sesuai standar RAG)
            # Potongan 800 karakter dengan overlap 100 agar konteks tidak terputus
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            
            # 3. EMBEDDING & VECTOR STORE (FAISS)
            vector_db = FAISS.from_documents(chunks, embeddings)
            print("RAG System Initialized Successfully")
        except Exception as e:
            print(f"Error during RAG Init: {e}")
            return None
    return vector_db

@app.get("/get-info")
async def get_info(artefak: str, lang: str = "id"):
    """Endpoint untuk deskripsi awal (Initial Scan)"""
    db = get_vector_db()
    if not db:
        raise HTTPException(status_code=500, detail="Database PDF gagal dimuat.")
    
    # Bersihkan label YOLO: kandean_dulang -> kandean dulang
    clean_name = artefak.replace("_", " ")
    query = f"Jelaskan secara mendalam tentang artefak {clean_name}"
    
    return await process_rag_request(query, clean_name, lang)

@app.get("/chat")
async def chat(query: str, artefak: str, lang: str = "id"):
    """Endpoint untuk tanya jawab lanjutan (Chat)"""
    db = get_vector_db()
    if not db:
        raise HTTPException(status_code=500, detail="Database PDF gagal dimuat.")
    
    clean_name = artefak.replace("_", " ")
    full_query = f"Pertanyaan user tentang {clean_name}: {query}"
    
    return await process_rag_request(full_query, clean_name, lang)

async def process_rag_request(user_query: str, artifact_name: str, lang: str):
    db = get_vector_db()
    
    # 1. RETRIEVAL (Similarity Search mencari chunk PDF yang paling cocok)
    docs = db.similarity_search(user_query, k=3)
    context = "\n".join([d.page_content for d in docs])

    # 2. PROMPT ENGINEERING
    if lang.lower() == "en":
        system_msg = "You are a Toraja cultural expert. Answer based on the provided context."
        prompt = f"Context:\n{context}\n\nArtifact: {artifact_name}\nQuestion: {user_query}\nAnswer in professional English:"
    else:
        system_msg = "Anda adalah ahli budaya Toraja. Jawab hanya berdasarkan konteks yang diberikan."
        prompt = f"Konteks:\n{context}\n\nArtefak: {artifact_name}\nPertanyaan: {user_query}\nJawab dalam Bahasa Indonesia formal yang menarik:"

    # 3. GENERATION (Llama 3.1)
    try:
        response = llm.invoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ])
        return {"description": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

# Untuk testing lokal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
