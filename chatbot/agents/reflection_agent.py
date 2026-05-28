"""
agents/reflection_agent.py — Self-Check / Reflection Agent

AGENTIC PATTERN: Reflection Pattern
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reflection pattern: Agent kendi çıktısını veya mevcut state'i
"dışarıdan bakarak" değerlendirir; eksiklik veya tutarsızlık
varsa bunu işaretler ve ilgili adıma geri döner.

Bu agent, onay ekranı gösterilmeden önce çalışır:
  - Tüm zorunlu slot'lar dolu mu?
  - Kefil zorunluysa TCKN var mı?
  - Çapraz kurallar hâlâ tutarlı mı? (örn: fatura değişince oran)

Neden Reflection önemli?
  - Slot filling sırasında oluşabilecek edge case'leri yakalar
  - LLM'e "güvenme, kontrol et" felsefesini uygular
  - Başvuru DB'ye hatalı kayıt yazılmasını önler
"""
import json
import time
from langchain_core.messages import AIMessage
from llm.factory import get_router_llm
from llm.prompts import REFLECTION_SYSTEM
from graph.state import ConversationState
from observability.audit_logger import log_llm_call
from config import NEW_CAR_GUARANTOR_THRESHOLD

# Ham slot adlarını müşteriye gösterilecek dostça etiketlere çevir
SLOT_LABELS = {
    "invoice_amount": "Proforma fatura tutarı",
    "vehicle_model": "Araç modeli",
    "requested_amount_new": "İstenen finansman tutarı",
    "guarantor_tckn": "Kefil TC kimlik numarası",
    "kasko_value": "Kasko değeri",
    "vehicle_age": "Araç yaşı",
    "requested_amount_used": "İstenen finansman tutarı",
    "seller_tckn": "Satıcı TC kimlik numarası",
}

# Geçersiz veya eksik slot için yeniden sorulacak sorular
SLOT_QUESTIONS = {
    "invoice_amount": "Proforma fatura tutarı nedir? (TL olarak)",
    "vehicle_model": "Aracın marka ve modelini belirtir misiniz? (örn: Toyota Corolla)",
    "requested_amount_new": "Ne kadar finansman almak istiyorsunuz? (TL olarak)",
    "guarantor_tckn": "Kefil TC kimlik numarası nedir?",
    "kasko_value": "Aracın güncel kasko değeri nedir? (TL olarak)",
    "vehicle_age": "Aracın model yılı veya yaşı nedir? (örn: 2021 model veya 3 yaşında)",
    "requested_amount_used": "Ne kadar finansman almak istiyorsunuz? (TL olarak)",
    "seller_tckn": "Satıcının TC kimlik numarası nedir? (Bilmiyorsanız 'geç' yazabilirsiniz)",
}


def run_reflection(state: ConversationState) -> dict:
    """
    Reflection node fonksiyonu.

    Returns:
        complete=True → recap'e geç
        complete=False → eksik/geçersiz alana geri dön
    """
    vehicle_type = state.get("vehicle_type")

    # Deterministik kontroller (LLM çağrısı yapmadan)
    issues = []        # kural ihlali mesajları (dostça metin)
    missing = []       # hiç doldurulmamış (None) slot'lar
    reset_slots = []   # dolu ama geçersiz → temizlenip yeniden sorulacak slot'lar

    if vehicle_type == "new":
        if not state.get("invoice_amount"):
            missing.append("invoice_amount")
        if not state.get("vehicle_model"):
            missing.append("vehicle_model")
        if not state.get("requested_amount_new"):
            missing.append("requested_amount_new")
        # Kefil kontrolü
        if state.get("kefil_required") and not state.get("guarantor_tckn"):
            missing.append("guarantor_tckn")
        # Çapraz kural: fatura değiştiyse oran hâlâ geçerli mi?
        if state.get("invoice_amount") and state.get("requested_amount_new"):
            max_allowed = state["invoice_amount"] * 0.60
            if state["requested_amount_new"] > max_allowed:
                issues.append(
                    f"Finansman tutarı ({_fmt(state['requested_amount_new'])} TL), "
                    f"proforma tutarının %60'ını ({_fmt(max_allowed)} TL) aşıyor."
                )
                reset_slots.append("requested_amount_new")

    elif vehicle_type == "used":
        if not state.get("kasko_value"):
            missing.append("kasko_value")
        if state.get("vehicle_age") is None:
            missing.append("vehicle_age")
        if not state.get("requested_amount_used"):
            missing.append("requested_amount_used")
        # Çapraz kural
        if state.get("kasko_value") and state.get("requested_amount_used"):
            max_allowed = min(state["kasko_value"] * 0.40, 3_000_000)
            if state["requested_amount_used"] > max_allowed:
                issues.append(
                    f"Finansman tutarı ({_fmt(state['requested_amount_used'])} TL) limiti aşıyor. "
                    f"Maksimum: {_fmt(max_allowed)} TL."
                )
                reset_slots.append("requested_amount_used")

    if missing or issues:
        parts = list(issues)
        if missing:
            labels = [SLOT_LABELS.get(m, m) for m in missing]
            parts.append("Eksik bilgiler: " + ", ".join(labels))

        # Önce geçersiz değeri düzelt, yoksa ilk eksik alanı tamamlat
        target = reset_slots[0] if reset_slots else (missing[0] if missing else None)
        if target:
            question = SLOT_QUESTIONS.get(target, f"Lütfen {SLOT_LABELS.get(target, target).lower()} bilgisini girin.")
            parts.append(question)

        msg = "Başvuruyu kontrol ettim, düzeltmemiz gereken noktalar var:\n" + "\n".join(f"• {p}" for p in parts)

        updates = {
            "current_step": "reflection_issues_found",
            "messages": state["messages"] + [AIMessage(content=msg)],
            "errors": state["errors"] + issues,
            "edit_target_slot": target,
        }
        # Geçersiz slot'ları temizle ki intake yeniden sorabilsin
        for s in reset_slots:
            updates[s] = None
        # Yeniden sorulacak slot'u awaited olarak işaretle → kullanıcı cevabı parse edilir
        if target:
            updates["awaited_slot"] = target
        return updates

    # Her şey tamam
    return {"current_step": "reflection_passed"}


def _fmt(amount: float) -> str:
    return f"{int(amount):,}".replace(",", ".")
