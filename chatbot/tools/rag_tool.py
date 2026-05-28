"""
tools/rag_tool.py — RAG Retrieval Aracı

AGENTIC PATTERN: Tool Use + Long-Term Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAQ agent bu tool'u çağırarak ChromaDB'den ilgili chunk'ları getirir.
Tool; embed et → ara → döndür işlemini kapsüller.
FAQ agent'ı sadece READ yetkisine sahiptir; slot okuyamaz/yazamaz.
"""
from langchain_core.tools import tool
from memory.vector_store import search_faq
from config import RAG_TOP_K


@tool
def retrieve_faq(query: str, top_k: int = RAG_TOP_K) -> dict:
    """
    Kullanıcı sorusuna en alakalı SSS içeriklerini getirir.

    Args:
        query: Kullanıcının sorusu
        top_k: Kaç chunk getirileceği (varsayılan config'den)

    Returns:
        {"chunks": [{"text": str, "title": str, "distance": float}],
         "context": str}  # LLM'e doğrudan verilecek birleşik metin
    """
    chunks = search_faq(query, top_k=top_k)

    if not chunks:
        return {
            "chunks": [],
            "context": "Bu konuda SSS dokümanında bilgi bulunamadı.",
        }

    # LLM'e verilecek context metnini oluştur
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[{i}] {chunk['title']}\n{chunk['text']}")

    return {
        "chunks": chunks,
        "context": "\n\n".join(context_parts),
    }
