import os
import json
from fastapi import FastAPI, HTTPException
import httpx # Library baru untuk memanggil Groq

app = FastAPI()

# Memuat "database" JSON kita
try:
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
except FileNotFoundError:
    knowledge_base = {}

# Ambil Groq API Key dari "rahasia" di Vercel
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.get("/get-info")
async def get_artifact_info(artefak: str):
    # Tahap 1: Retrieval (Mengambil Konteks)
    context = knowledge_base.get(artefak)
    
    if not context:
        raise HTTPException(status_code=404, detail=f"Konteks untuk '{artefak}' tidak ditemukan di database.")

    if not GROQ_API_KEY:
         raise HTTPException(status_code=500, detail="Kunci API Groq belum diatur di server Vercel.")

    # Tahap 2: Augmentation (Membuat Prompt yang Diperkaya)
    prompt = f"Berdasarkan konteks berikut: '{context}'. Jelaskan secara menarik dan informatif dalam format paragraf apa itu {artefak}."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama-3.1-8b-instant",
    }

    # Tahap 3: Generation (Memanggil Groq)
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GROQ_API_URL, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            groq_response = response.json()
            llm_content = groq_response["choices"][0]["message"]["content"]
            return {"artifact_name": artefak, "description": llm_content}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Terjadi error saat memproses respons dari Groq: {e}")
