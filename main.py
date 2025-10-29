
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from hashlib import sha256
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI(title="C-FROG Cloud API")

@app.get("/")
def root():
    return {"message": "Welcome to C-FROG Cloud API"}

TENANTS = {}
SECRET = os.environ.get("C_FROG_SECRET", "dev-secret")

def ensure_tenant(t):
    if t not in TENANTS:
        TENANTS[t] = {"docs": [], "tfidf": None, "mat": None}

def build_index(t):
    docs = TENANTS[t]["docs"]
    if not docs: return
    v = TfidfVectorizer(max_features=5000)
    m = v.fit_transform(docs)
    TENANTS[t]["tfidf"], TENANTS[t]["mat"] = v, m

def retrieve(t, q, k=3):
    v, m = TENANTS[t]["tfidf"], TENANTS[t]["mat"]
    if v is None: return []
    qv = v.transform([q])
    sims = cosine_similarity(qv, m)[0]
    idx = sims.argsort()[::-1][:k]
    return [(TENANTS[t]["docs"][i], float(sims[i])) for i in idx]

class Ingest(BaseModel):
    tenant_id: str
    documents: List[str]

class Ask(BaseModel):
    tenant_id: str
    query: str

@app.post("/ingest")
def ingest(req: Ingest):
    ensure_tenant(req.tenant_id)
    TENANTS[req.tenant_id]["docs"].extend(req.documents)
    build_index(req.tenant_id)
    return {"ok": True, "count": len(TENANTS[req.tenant_id]["docs"])}

@app.post("/ask")
def ask(req: Ask):
    ensure_tenant(req.tenant_id)
    hits = retrieve(req.tenant_id, req.query)
    if not hits:
        return {"answer": "No relevant data found", "policy_ok": True}
    answer = hits[0][0]
    proof = sha256((answer + SECRET).encode()).hexdigest()
    return {"answer": answer, "proof": proof, "policy_ok": True}
