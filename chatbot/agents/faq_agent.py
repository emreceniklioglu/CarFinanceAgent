"""
agents/faq_agent.py — FAQ / RAG Agent

AGENTIC PATTERN: RAG (Retrieval-Augmented Generation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG pattern'de model yeni bilgi üretmez; mevcut dokümanı
kullanarak soruyu cevaplar. Bu sayede:
- Halüsinasyon riski minimum (sadece getirilen chunk'tan cevap)
- Cevap kaynağı şeffaf (kaynak chunk gösterilir)
- Doküman güncellenince otomatik güncel cevap

Akış:
  1. retrieve_faq tool'u → ChromaDB'den top-k chunk
  2. Chunk'lar prompt'a context olarak eklenir
  3. Büyük model cevap üretir (sadece context'ten)
  4. Output Guard: kaynak chunk zorunlu
  5. State snapshot'tan kaldığı yere dönülür

SCOPED ACCESS: FAQ agent sadece VectorDB read yetkisine sahip.
Slot okuyamaz/yazamaz — başka müşterinin verisi sızmaz.
"""
import time
from langchain_core.messages import AIMessage
from tools.rag_tool import retrieve_faq
from llm.factory import get_main_llm, get_llm
from llm.prompts import FAQ_SYSTEM
from graph.state import ConversationState
from guardrails.output_guard import check_output
from observability.audit_logger import log_llm_call, log_tool_call


def run_faq_agent(state: ConversationState) -> dict:
    """
    FAQ node fonksiyonu.

    Müşterinin sorusunu RAG ile yanıtlar, ardından
    kaldığı adıma devam etmesini önerir.
    """
    last_message = state["messages"][-1]
    user_query = last_message.content if hasattr(last_message, "content") else str(last_message)

    # ── Adım 1: Retrieval ─────────────────────────────────────────────────────
    retrieval_result = retrieve_faq.invoke({"query": user_query})
    chunks = retrieval_result["chunks"]
    context = retrieval_result["context"]

    log_tool_call(
        session_id=state["session_id"],
        agent="faq_agent",
        tool_name="retrieve_faq",
        args={"query": user_query[:100]},
        result={"chunk_count": len(chunks)},
        success=True,
    )

    # ── Adım 2: LLM ile cevap üret ────────────────────────────────────────────
    llm = get_main_llm()
    system_with_context = FAQ_SYSTEM.format(context=context)

    faq_messages = [
        {"role": "system", "content": system_with_context},
        {"role": "user", "content": user_query},
    ]

    start = time.time()
    response = llm.invoke(faq_messages)
    # Gemini RECITATION/SAFETY filtresinde boş içerik dönebilir; farklı sampling
    # ile bir kez daha dene (sıcaklık > 0 genelde filtreyi aşar).
    if not (response.content or "").strip():
        response = get_llm(size="large", temperature=0.5).invoke(faq_messages)
    latency = int((time.time() - start) * 1000)

    log_llm_call(
        session_id=state["session_id"],
        agent="faq_agent",
        model=llm.model,
        latency_ms=latency,
        tokens_in=len(system_with_context.split()) + len(user_query.split()),
        tokens_out=len(response.content.split()),
        prompt=[
            {"role": "system", "content": system_with_context},
            {"role": "user", "content": user_query},
        ],
        response=response.content,
    )

    # ── Adım 3: Output Guard ──────────────────────────────────────────────────
    guard_result = check_output(
        text=response.content,
        session_id=state["session_id"],
        is_faq_response=True,
        source_chunks=chunks,
    )

    if not guard_result["allowed"]:
        answer = "Bu konuda şu anda yardımcı olamıyorum. Lütfen şubemizi arayın."
    else:
        answer = guard_result["cleaned_text"]

    # ── Adım 4: Resume mesajı ekle ────────────────────────────────────────────
    pending_question = _get_pending_question(state)
    resume_hint = f"\n\n---\n*Kaldığımız yere dönelim: {pending_question}*" if pending_question else ""

    full_answer = answer + resume_hint

    return {
        "current_step": "faq_answered",
        "messages": state["messages"] + [AIMessage(content=full_answer)],
        # Snapshot korunur; bir sonraki turda supervisor yeniden çalışır
    }


# Bekleyen soru metinleri — slot bazlı
_PENDING_SLOT_QUESTIONS = {
    "kasko_value": "Aracın güncel kasko değeri nedir? (TL olarak)",
    "vehicle_age": "Aracın model yılı veya yaşı nedir? (örn: 2021 model veya 3 yaşında)",
    "requested_amount_used": "Ne kadar finansman almak istiyorsunuz? (TL olarak)",
    "seller_tckn": "Satıcının TC kimlik numarası nedir? (Bilmiyorsanız 'geç' yazabilirsiniz)",
    "invoice_amount": "Aracın proforma fatura tutarı nedir? (TL olarak)",
    "vehicle_model": "Hangi marka ve model aracı düşünüyorsunuz?",
    "requested_amount_new": "Ne kadar finansman almak istiyorsunuz? (TL olarak)",
    "guarantor_tckn": "Kefilinizin TC kimlik numarası nedir?",
}


def _get_pending_question(state: ConversationState) -> str | None:
    """FAQ'ten sonra kullanıcının döneceği bekleyen soruyu üretir."""
    awaited = state.get("awaited_slot")
    if awaited and awaited in _PENDING_SLOT_QUESTIONS:
        return _PENDING_SLOT_QUESTIONS[awaited]
    # Henüz araç türü seçilmemişse baştaki soruya dön
    if not state.get("vehicle_type"):
        return "Yeni araç mı, 2. el araç mı düşünüyorsunuz?"
    return None
