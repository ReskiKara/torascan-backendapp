from fastapi import FastAPI, HTTPException
import json

app = FastAPI()

# Memuat "database" JSON saat aplikasi dimulai
try:
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
except FileNotFoundError:
    # Jika file tidak ada, buat database kosong untuk menghindari crash
    knowledge_base = {}

@app.get("/get-info")
async def get_artifact_info(artefak: str):
    description = knowledge_base.get(artefak)
    
    if not description:
        # Jika artefak tidak ditemukan di "database" kita
        raise HTTPException(status_code=404, detail="Informasi untuk artefak ini tidak ditemukan.")
        
    return {"artifact_name": artefak, "description": description}
