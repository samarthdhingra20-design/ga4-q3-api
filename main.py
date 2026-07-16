from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

import json
import config
import re
import pandas as pd
import numpy as np

# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(title="Grounded QA API")

# ==========================================================
# Load Q4 Dataset
# ==========================================================

DOCUMENTS = pd.read_csv("documents.csv")

with open("embeddings.json", "r") as f:
    EMBEDDINGS = json.load(f)

with open("reranker_scores.json", "r") as f:
    RERANKER = json.load(f)

print("=" * 50)
print("Q4 DATA LOADED")
print("=" * 50)
print("Documents :", len(DOCUMENTS))
print("Embeddings :", len(EMBEDDINGS))
print("Queries :", len(RERANKER))

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


class VectorSearchRequest(BaseModel):
    query_id: str
    query_vector: List[float]
    top_k: int
    rerank_top_n: int
    filter: Dict[str, Any]


# ==========================================================
# Cosine Similarity
# ==========================================================

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


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
# Grounded Answer Endpoint
# ==========================================================

@app.post("/grounded-answer")
async def grounded_answer(body: QARequest):

    question = body.question.strip().lower()

    # ------------------------------------------------------
    # Empty input
    # ------------------------------------------------------

    if question == "" or len(body.chunks) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.1,
            "answerable": False
        }

    # ------------------------------------------------------
    # Extract keywords
    # ------------------------------------------------------

    stop_words = {
        "what","when","where","who","which","why","how",
        "is","was","are","were","be","been",
        "the","a","an","of","to","in","on","for",
        "and","or","did","does","do","with","from",
        "by","at","tell","me","about","explain",
        "give","this","that","these","those"
    }

    keywords = [
        w
        for w in re.findall(r"\w+", question)
        if len(w) > 2 and w not in stop_words
    ]

    best_chunk = None
    best_score = 0

    # ------------------------------------------------------
    # Score every chunk
    # ------------------------------------------------------

    for chunk in body.chunks:

        text = " ".join(chunk.text.strip().split())
        text_lower = text.lower()

        words = set(re.findall(r"\w+", text_lower))

        score = 0

        # keyword matches
        for word in keywords:
            if word in words:
                score += 2

        # year bonus
        if (
            ("when" in question or "year" in question)
            and re.search(r"\b(19|20)\d{2}\b", text)
        ):
            score += 2

        # release/open-source bonus
        release_words = [
            "release",
            "released",
            "open-source",
            "open-sourced",
            "launch",
            "launched",
            "introduced"
        ]

        if any(word in text_lower for word in release_words):
            score += 2

        if score > best_score:
            best_score = score
            best_chunk = chunk

    # ------------------------------------------------------
    # Anti-hallucination
    # ------------------------------------------------------

    if best_chunk is None or best_score == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.1,
            "answerable": False
        }

    confidence = min(
        0.95,
        0.55 + 0.05 * best_score
    )

    return {
        "answer": " ".join(best_chunk.text.strip().split()),
        "citations": [best_chunk.chunk_id],
        "confidence": round(confidence, 2),
        "answerable": True
    }
# ==========================================================
# Vector Search Endpoint (Q4)
# ==========================================================

@app.post("/vector-search")
async def vector_search(body: VectorSearchRequest):

    query_id = body.query_id
    query_vector = np.array(body.query_vector, dtype=np.float32)
    top_k = body.top_k
    rerank_top_n = body.rerank_top_n
    filters = body.filter

    # Validate vector length
    if len(query_vector) != 100:
        return {"matches": []}

    # ------------------------------------------------------
    # Step 1 : Metadata Filtering
    # ------------------------------------------------------

    filtered_docs = []

    for _, doc in DOCUMENTS.iterrows():

        keep = True

        for field, condition in filters.items():

            # Unknown column
            if field not in DOCUMENTS.columns:
                keep = False
                break

            if not isinstance(condition, dict):

                if doc[field] != condition:
                    keep = False
                    break

            else:

                if "gte" in condition and doc[field] < condition["gte"]:
                    keep = False
                    break

                if "lte" in condition and doc[field] > condition["lte"]:
                    keep = False
                    break

                if "in" in condition and doc[field] not in condition["in"]:
                    keep = False
                    break

        if keep:
            filtered_docs.append(doc)

    if len(filtered_docs) == 0:
        return {"matches": []}

    # ------------------------------------------------------
    # Step 2 : Cosine Similarity
    # ------------------------------------------------------

    scored = []

    for doc in filtered_docs:

        doc_id = doc["doc_id"]

        if doc_id not in EMBEDDINGS:
            continue

        similarity = cosine_similarity(
            query_vector,
            EMBEDDINGS[doc_id]
        )

        scored.append({
            "doc_id": doc_id,
            "similarity": similarity
        })

    scored.sort(
        key=lambda x: (
            -x["similarity"],
            x["doc_id"]
        )
    )

    top_docs = scored[:top_k]

    # ------------------------------------------------------
    # Step 3 : Re-ranking
    # ------------------------------------------------------

    scores = RERANKER.get(query_id, {})

    reranked = []

    for doc in top_docs:

        reranked.append({
            "doc_id": doc["doc_id"],
            "score": scores.get(doc["doc_id"], -9999)
        })

    reranked.sort(
        key=lambda x: (
            -x["score"],
            x["doc_id"]
        )
    )

    matches = [
        doc["doc_id"]
        for doc in reranked[:rerank_top_n]
    ]

    return {
        "matches": matches
    }