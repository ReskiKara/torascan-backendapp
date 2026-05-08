# =========================
# RAG PROCESS
# =========================
async def process_rag(user_query: str, artifact_name: str, lang: str):

    db = get_vector_db()

    # =========================
    # RETRIEVAL
    # AMBIL BEBERAPA CHUNK
    # =========================
    docs = db.similarity_search(
        artifact_name,
        k=5
    )

    # =========================
    # FILTER CHUNK RELEVAN
    # =========================
    relevant_chunks = []

    for doc in docs:

        # Fokus cek nama artefak
        # di awal chunk
        first_part = doc.page_content.lower()[:150]

        if artifact_name.lower() in first_part:

            relevant_chunks.append(
                doc.page_content
            )

    # =========================
    # FALLBACK
    # =========================
    if not relevant_chunks:

        context = docs[0].page_content

    else:

        context = "\n".join(relevant_chunks)

    print("RETRIEVED CONTEXT:")
    print(context)

    # =========================
    # PROMPT
    # =========================
    if lang.lower() == "en":

        system_msg = f"""
You are a Toraja cultural information system.

FOCUS ONLY ON:
{artifact_name}

RULES:
- Answer ONLY based on retrieval context.
- Do not mix other artifacts.
- Do not add outside information.
- Maximum 2 paragraphs.
- Use concise and formal language.
"""

        prompt = f"""
ANSWER CONTRACT:

1. Artifact Name:
{artifact_name}

2. Use ONLY this context:
{context}

TASK:
Explain the artifact {artifact_name}.
If the context discusses other artifacts,
IGNORE them and use only relevant information.

Answer in concise professional English.
"""

    else:

        system_msg = f"""
Anda adalah sistem informasi artefak budaya Toraja.

FOKUS HANYA PADA:
{artifact_name}

ATURAN:
- Jawab HANYA berdasarkan context retrieval.
- Jangan mencampur artefak lain.
- Jangan menambahkan informasi luar.
- Jawaban maksimal 2 paragraf.
- Gunakan bahasa formal dan ringkas.
"""

        prompt = f"""
KONTRAK JAWABAN:

1. Nama artefak yang harus dijelaskan:
{artifact_name}

2. Gunakan HANYA konteks berikut:
{context}

TUGAS:
Jelaskan artefak {artifact_name}.

Jika konteks membahas artefak lain,
ABAIAKAN dan gunakan hanya informasi
yang relevan dengan {artifact_name}.

Jawaban maksimal 2 paragraf.
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
