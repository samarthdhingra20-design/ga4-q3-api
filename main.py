from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import httpx
import json
import config

# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(title="Grounded QA API")

# ==========================================================
# Enable CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Request Models
# ==========================================================

class Chunk(BaseModel):
    chunk_id: str
    text: str


class QARequest(BaseModel):
    question: str
    chunks: List[Chunk]


# ==========================================================
# AIPipe Configuration
# ==========================================================

HEADERS = {
    "Authorization": f"Bearer {config.AIPIPE_TOKEN}",
    "Content-Type": "application/json"
}

# ==========================================================
# Chat Function
# ==========================================================

async def chat(prompt: str):

    body = {
        "model": config.TEXT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {
            "type": "json_object"
        }
    }

    async with httpx.AsyncClient(timeout=60) as client:

        response = await client.post(
            f"{config.AIPIPE_BASE}/chat/completions",
            headers=HEADERS,
            json=body
        )

        print("=" * 60)
        print("STATUS:", response.status_code)
        print("RESPONSE:")
        print(response.text)
        print("=" * 60)

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]


# ==========================================================
# Root Endpoint
# ==========================================================

@app.get("/")
async def root():

    return {
        "status": "API Running",
        "email": config.EMAIL
    }


# ==========================================================
# Test Endpoint
# ==========================================================

@app.get("/test")
async def test():

    prompt = """
Return ONLY this JSON.

{
    "message":"Hello from GPT"
}
"""

    try:

        response = await chat(prompt)

        return json.loads(response)

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ==========================================================
# Grounded Answer Endpoint
# ==========================================================

@app.post("/grounded-answer")
async def grounded_answer(body: QARequest):

    question = body.question.strip()

    chunks = [
        chunk.model_dump()
        for chunk in body.chunks
    ]

    # ------------------------------------------------------

    if question == "" or len(chunks) == 0:

        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.1,
            "answerable": False
        }

    # ------------------------------------------------------

    prompt = f"""
You are a highly reliable Grounded QA API.

Rules:

1. Use ONLY the provided chunks.

2. Never use outside knowledge.

3. If the answer cannot be found in the chunks, return EXACTLY

{{
    "answer":"I don't know",
    "citations":[],
    "confidence":0.1,
    "answerable":false
}}

4. Otherwise return

{{
    "answer":"...",
    "citations":["C1"],
    "confidence":0.95,
    "answerable":true
}}

5. Return ONLY JSON.

QUESTION

{question}

CHUNKS

{json.dumps(chunks, indent=2)}
"""

    # ------------------------------------------------------

    try:

        response = await chat(prompt)

        result = json.loads(response)

    except Exception as e:

        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.1,
            "answerable": False,
            "error": str(e)
        }

    # ------------------------------------------------------
    # Validate Citation IDs
    # ------------------------------------------------------

    valid_ids = {
        chunk["chunk_id"]
        for chunk in chunks
    }

    citations = []

    for cid in result.get("citations", []):

        if cid in valid_ids:
            citations.append(cid)

    # ------------------------------------------------------

    if result.get("answerable", False) is False:

        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.1,
            "answerable": False
        }

    # ------------------------------------------------------

    confidence = float(
        result.get(
            "confidence",
            0.9
        )
    )

    confidence = max(0.0, min(1.0, confidence))

    return {

        "answer": result.get(
            "answer",
            "I don't know"
        ),

        "citations": citations,

        "confidence": confidence,

        "answerable": True
    }