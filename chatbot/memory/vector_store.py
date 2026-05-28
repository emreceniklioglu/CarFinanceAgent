"""
memory/vector_store.py — Jina Embedding + ChromaDB RAG Altyapısı

AGENTIC PATTERN: Long-Term Memory (RAG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uzun vadeli hafıza = SSS dokümanının vektörleştirilmiş hali.
Uygulama her başladığında FAQ dokümanı chunk'lara bölünür,
Jina API ile embed edilir ve ChromaDB'ye kaydedilir.

ChromaDB, veritabanını diske persist eder (data/chroma/).
Uygulama yeniden başlatılınca aynı vektörler tekrar kullanılır
(re-indeksleme yapılmaz).

Jina AI free tier: https://jina.ai/embeddings/
  - Model: jina-embeddings-v3
  - 1M token/ay ücretsiz
  - 8192 token context window
  - Türkçe dahil çok dilli destek
"""
from __future__ import annotations
import os
import re
import httpx
import chromadb
from config import (
    CHROMA_PERSIST_DIR,
    FAQ_COLLECTION_NAME,
    JINA_API_KEY,
    JINA_API_URL,
    JINA_EMBEDDING_MODEL,
    RAG_TOP_K,
)

FAQ_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faq.md")

# ChromaDB istemcisi (singleton)
_client: chromadb.PersistentClient | None = None
_collection = None


def _get_client():
    global _client, _collection
    if _client is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(FAQ_COLLECTION_NAME)
    return _client, _collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Jina AI API ile metinleri vektörleştirir.

    Jina'nın 'retrieval.passage' görevi embedding üretim içindir.
    Sorgu zamanında 'retrieval.query' kullanılır (asimetrik retrieval).
    """
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY tanımlı değil. .env dosyasını kontrol edin.")

    response = httpx.post(
        JINA_API_URL,
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
        json={"model": JINA_EMBEDDING_MODEL, "task": "retrieval.passage", "input": texts},
        timeout=30.0,
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]


def embed_query(query: str) -> list[float]:
    """
    Sorgu metnini vektörleştirir (passage'dan farklı görev).
    Asimetrik retrieval: soru ve cevap farklı embedding uzayında.
    """
    response = httpx.post(
        JINA_API_URL,
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
        json={"model": JINA_EMBEDDING_MODEL, "task": "retrieval.query", "input": [query]},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def _chunk_faq(faq_text: str) -> list[dict]:
    """
    FAQ dokümanını başlık + içerik çiftlerine böler.
    ### başlığı bir chunk sınırıdır.

    Returns:
        [{"id": str, "text": str, "title": str}, ...]
    """
    chunks = []
    sections = re.split(r"\n### ", faq_text)

    for i, section in enumerate(sections):
        if not section.strip():
            continue
        lines = section.strip().split("\n", 1)
        title = lines[0].strip("# ").strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        if content:
            chunks.append({
                "id": f"faq_{i:03d}",
                "text": f"Soru: {title}\nCevap: {content}",
                "title": title,
            })
    return chunks


def index_faq(force_reindex: bool = False):
    """
    FAQ dokümanını okuyup ChromaDB'ye indeksler.
    force_reindex=False → zaten indekslenmişse atla.
    """
    _, collection = _get_client()

    # Zaten indekslenmişse atla
    if not force_reindex and collection.count() > 0:
        print(f"[VectorStore] {collection.count()} chunk zaten indeksli, atlanıyor.")
        return

    with open(FAQ_PATH, encoding="utf-8") as f:
        faq_text = f.read()

    chunks = _chunk_faq(faq_text)
    print(f"[VectorStore] {len(chunks)} chunk indeksleniyor...")

    # Jina API limitleri için batch halinde embed et
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)

        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings,
            metadatas=[{"title": c["title"]} for c in batch],
        )

    print(f"[VectorStore] İndeksleme tamamlandı. Toplam: {collection.count()} chunk")


def search_faq(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """
    Sorguya en alakalı FAQ chunk'larını döndürür.

    Returns:
        [{"text": str, "title": str, "distance": float}, ...]
    """
    _, collection = _get_client()

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "title": meta["title"], "distance": dist})

    return chunks
