import os
import json
from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI()

# ======================
# LOAD DATABASE JSON
# ======================
try:
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
except FileNotFoundError:
    knowledge_base = {}

# ======================
# API CONFIG
# ======================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# ======================
# ENDPOINT
# ======================
@app.get("/get-info")
async def get_artifact_info(artefak: str):

    # ======================
    # 1. RETRIEVAL
    # ======================
    context = knowledge_base.get(artefak.lower())

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
    # 2. PROMPT
    # ======================
    prompt = f"""
Berdasarkan konteks berikut:
{context}

Jelaskan artefak {artefak}  secara informatif dan menarik dalam bentuk paragraf.

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
        "max_tokens": 500
    }

    # ======================
    # 3. CALL GROQ API
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

            groq_response = response.json()

            if "choices" not in groq_response:
                raise HTTPException(
                    status_code=500,
                    detail=f"Response tidak valid: {groq_response}"
                )

            llm_content = groq_response["choices"][0]["message"]["content"]

            return {
                "artifact_name": artefak,
                "description": llm_content
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Terjadi error saat memproses respons dari Groq: {e}"
            )
