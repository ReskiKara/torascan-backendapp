import os
import json
from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI()

# Load database JSON
try:
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
except FileNotFoundError:
    knowledge_base = {}

# Ambil API Key dari Vercel
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


@app.get("/get-info")
async def get_artifact_info(artefak: str):
    # ======================
    # 1. RETRIEVAL
    # ======================
    context = knowledge_base.get(artefak)

    if not context:
        raise HTTPException(
            status_code=404,
            detail=f"Konteks untuk '{artefak}' tidak ditemukan di database."
        )

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Kunci API Groq belum diatur di server Vercel."
        )

    # ======================
    # 2. AUGMENTATION (PROMPT)
    # ======================
    prompt = f"""
Berdasarkan konteks berikut:
{context}

Jelaskan artefak {artefak} dalam 2 paragraf singkat.

Ketentuan:
- Artefak berasal dari Toraja
- Gunakan bahasa Indonesia formal
- Jangan menyebut budaya lain
- Gunakan hanya informasi dari konteks
- Jangan mengarang informasi
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": "llama-3.1-8b-instant",
        "max_tokens": 150  # 🔥 WAJIB biar cepat
    }

    # ======================
    # 3. GENERATION (CALL API)
    # ======================
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                GROQ_API_URL,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()

            groq_response
