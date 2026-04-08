import os
import json
from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI()

# ======================
# LOAD DATABASE JSON
# ======================
try:
    # Memastikan file dibaca dengan path yang benar di Vercel
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "knowledge_base.json")
    with open(file_path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
except Exception as e:
    knowledge_base = {}
    print(f"Error loading database: {e}")

# ======================
# API CONFIG
# ======================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ======================
# ENDPOINT
# ======================
@app.get("/get-info")
async def get_artifact_info(artefak: str, lang: str = "id"):
    """
    Endpoint untuk mendapatkan informasi artefak.
    Parameter 'lang' menerima 'id' (Indonesia) atau 'en' (English).
    """

    # 1. RETRIEVAL (Ambil konteks dari JSON)
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

    # 2. KONFIGURASI PROMPT BERDASARKAN BAHASA
    if lang.lower() == "en":
        system_instruction = "You are a cultural expert on Toraja artifacts."
        user_prompt = f"""
Based on the following context:
{context}

Explain the artifact '{artefak}' informatively and engagingly in paragraphs.

Rules:
- The artifact is from Toraja, South Sulawesi, Indonesia.
- Use formal and professional English.
- Use only information provided in the context.
- Do not mention other cultures or make up information.
- Format the output clearly.
"""
    else:
        # Default Bahasa Indonesia
        system_instruction = "Anda adalah ahli budaya artefak Toraja."
        user_prompt = f"""
Berdasarkan konteks berikut:
{context}

Jelaskan artefak {artefak} secara informatif dan menarik dalam bentuk paragraf.

Ketentuan:
- Artefak berasal dari Toraja, Sulawesi Selatan.
- Gunakan Bahasa Indonesia formal yang baik dan benar.
- Gunakan hanya informasi dari konteks yang disediakan.
- Jangan menyebut budaya lain atau mengarang informasi.
- Format hasil dalam bentuk paragraf yang mudah dibaca.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        "model": "llama-3.1-8b-instant",
        "max_tokens": 800,
        "temperature": 0.5 # Menjaga respon tetap konsisten dan tidak terlalu kreatif/halusinasi
    }

    # 3. CALL GROQ API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                GROQ_API_URL,
                json=payload,
                headers=headers,
                timeout=45.0
            )

            response.raise_for_status()
            groq_response = response.json()

            if "choices" not in groq_response:
                raise HTTPException(
                    status_code=500,
                    detail=f"Response tidak valid dari AI."
                )

            llm_content = groq_response["choices"][0]["message"]["content"]

            return {
                "artifact_name": artefak,
                "language": lang,
                "description": llm_content
            }

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Server AI (Groq) terlalu lama merespon.")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Terjadi error saat memproses respons dari Groq: {str(e)}"
            )
