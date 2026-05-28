"""
graph/nodes.py — LangGraph Node Fonksiyonları

AGENTIC PATTERN: State Machine — Nodes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Her node, state'i alır, bir işlem yapar ve güncellenmiş
state parçasını döndürür. LangGraph bu parçaları mevcut
state ile birleştirerek yeni state oluşturur.

Not: Bu dosya agent modüllerini çağıran ince bir sarmalayıcıdır.
İş mantığı agent dosyalarında; bağlantı mantığı burada.
"""
import time
from langchain_core.messages import AIMessage, HumanMessage
from graph.state import ConversationState
from guardrails.input_guard import check_input, get_block_message
from guardrails.output_guard import check_output
from agents.supervisor import run_supervisor
from agents.intake_new import run_intake_new
from agents.intake_used import run_intake_used
from agents.faq_agent import run_faq_agent
from agents.validation_agent import validate_new_car_state, validate_used_car_state
from agents.reflection_agent import run_reflection
from agents.submission_agent import run_submission
from agents.crosssell_agent import run_crosssell
from observability.audit_logger import log_guard_event


def input_guard_node(state: ConversationState) -> dict:
    """
    L4 — Input Guard Node.
    Her turda ilk çalışan node; güvenlik geçmezse akış durur.
    """
    last_message = state["messages"][-1]
    user_text = last_message.content if hasattr(last_message, "content") else ""

    guard_result = check_input(user_text, state["session_id"])

    if not guard_result["allowed"]:
        block_msg = get_block_message()
        # flow_ended set etme — sadece bu mesajı engelle, session'ı öldürme;
        # bir sonraki normal mesajda kaldığı yerden devam edilebilsin.
        return {
            "current_step": state.get("current_step", "input_blocked"),
            "messages": state["messages"] + [AIMessage(content=block_msg)],
        }

    # PII maskelenmişse mesajı güncelle
    if guard_result["pii_detected"]:
        updated_messages = state["messages"][:-1] + [
            HumanMessage(content=guard_result["sanitized_text"])
        ]
        return {"messages": updated_messages, "current_step": "input_guard_passed"}

    return {"current_step": "input_guard_passed"}


def supervisor_node(state: ConversationState) -> dict:
    """Supervisor / Router Agent."""
    return run_supervisor(state)


def intake_new_node(state: ConversationState) -> dict:
    """Yeni Araç Slot Toplama."""
    return run_intake_new(state)


def intake_used_node(state: ConversationState) -> dict:
    """2. El Araç Slot Toplama."""
    return run_intake_used(state)


def faq_agent_node(state: ConversationState) -> dict:
    """FAQ / RAG Agent."""
    # FAQ moduna girerken snapshot kaydet
    if not state.get("faq_snapshot"):
        state = {**state, "faq_snapshot": {"step": state.get("current_step")}}
    return run_faq_agent(state)


def validation_new_node(state: ConversationState) -> dict:
    """Yeni Araç Doğrulama."""
    return validate_new_car_state(state)


def validation_used_node(state: ConversationState) -> dict:
    """2. El Araç Doğrulama."""
    return validate_used_car_state(state)


def reflection_node(state: ConversationState) -> dict:
    """Self-Check / Reflection."""
    return run_reflection(state)


_EDIT_SLOT_KEYWORDS_USED = {
    "kasko_value":          ["kasko"],
    "vehicle_age":          ["yaş", "model yılı", "yıl"],
    "requested_amount_used":["finansman", "tutar", "kredi", "miktar"],
    "seller_tckn":          ["satıcı tckn", "satıcı kimlik", "satıcının tc"],
}
_EDIT_SLOT_KEYWORDS_NEW = {
    "invoice_amount":       ["fatura", "proforma"],
    "vehicle_model":        ["model", "marka"],
    "requested_amount_new": ["finansman", "tutar", "kredi", "miktar"],
    "guarantor_tckn":       ["kefil", "garantör"],
}
_RECAP_SLOT_QUESTIONS = {
    "kasko_value":          "Aracın güncel kasko değeri nedir? (TL olarak)",
    "vehicle_age":          "Aracın model yılı veya yaşı nedir? (örn: 2021 model veya 3 yaşında)",
    "requested_amount_used":"Ne kadar finansman almak istiyorsunuz? (TL olarak)",
    "seller_tckn":          "Satıcının TC kimlik numarası nedir? (Bilmiyorsanız 'geç' yazabilirsiniz)",
    "invoice_amount":       "Aracın proforma fatura tutarı nedir? (TL olarak)",
    "vehicle_model":        "Hangi marka ve model aracı düşünüyorsunuz?",
    "requested_amount_new": "Ne kadar finansman almak istiyorsunuz? (TL olarak)",
    "guarantor_tckn":       "Kefilinizin TC kimlik numarası nedir?",
}


def _detect_edit_slot(lower: str, vehicle_type: str) -> str | None:
    keywords = _EDIT_SLOT_KEYWORDS_USED if vehicle_type == "used" else _EDIT_SLOT_KEYWORDS_NEW
    for slot, words in keywords.items():
        if any(w in lower for w in words):
            return slot
    return None


def recap_node(state: ConversationState) -> dict:
    """
    Onay Ekranı Node.
    Tüm slot'ları müşteriye gösterir; onay veya düzenleme bekler.
    """
    last_message = state["messages"][-1]
    user_text = last_message.content if hasattr(last_message, "content") else ""

    # Eğer son mesaj kullanıcıdan geldiyse → onay / düzenleme parse et
    if hasattr(last_message, "type") and last_message.type != "ai":
        lower = user_text.lower().strip()
        # Onay yalnızca tam "onaylıyorum" ile verilir — "evet/tamam" gibi ifadeler yetmez
        if lower == "onaylıyorum":
            return {"recap_confirmed": True, "current_step": "recap_confirmed"}

        # Araç türü değişikliği → tüm slotları sıfırla, baştan başla
        vehicle_type_words = ["araç türü", "arac turu", "araç tipi", "yeni araç", "2.el", "2. el", "ikinci el", "sıfır araç"]
        if any(w in lower for w in vehicle_type_words):
            return {
                "current_step": "start",
                "vehicle_type": None,
                "invoice_amount": None, "vehicle_model": None, "requested_amount_new": None,
                "guarantor_tckn": None, "kefil_required": False,
                "kasko_value": None, "vehicle_age": None, "requested_amount_used": None,
                "seller_tckn": None,
                "awaited_slot": None, "edit_target_slot": None, "recap_confirmed": False,
                "errors": [],
                "messages": state["messages"] + [AIMessage(content="Yeni araç mı, 2. el araç mı düşünüyorsunuz?")],
            }

        # Hangi slot güncellenmek isteniyor?
        target_slot = _detect_edit_slot(lower, state.get("vehicle_type"))
        if target_slot:
            question = _RECAP_SLOT_QUESTIONS.get(target_slot, "Yeni değeri giriniz.")
            return {
                "current_step": "recap_edit_requested",
                "recap_confirmed": False,
                "edit_target_slot": target_slot,
                "awaited_slot": target_slot,
                target_slot: None,
                "messages": state["messages"] + [AIMessage(content=question)],
            }

        # Düzenleme niyeti var ama hangi alan belli değil → sor
        edit_words = ["değiştir", "düzelt", "güncelle", "yanlış", "hata", "değiştirmek", "güncellemek"]
        if any(w in lower for w in edit_words):
            msg = "Hangi bilgiyi güncellemek istersiniz? (örn: kasko değeri, finansman tutarı, araç yaşı)"
            return {
                "current_step": "recap_shown",
                "messages": state["messages"] + [AIMessage(content=msg)],
            }

    # İlk kez Recap (veya son mesaj AI ise) → özet göster
    summary = _build_recap_message(state)
    return {
        "current_step": "recap_shown",
        "messages": state["messages"] + [AIMessage(content=summary)],
    }


def submission_node(state: ConversationState) -> dict:
    """Başvuru Kayıt Agent."""
    return run_submission(state)


def crosssell_node(state: ConversationState) -> dict:
    """HGS Çapraz Satış."""
    return run_crosssell(state)


def wait_node(state: ConversationState) -> dict:
    """
    Pasif bekleme node'u — kullanıcı cevabı bekliyor.
    current_step'e DOKUNMAZ: önceki node'un set ettiği "asking_xxx" değeri
    korunmalı ki bir sonraki turda intake hangi slot'u beklediğini bilsin.
    """
    return {}


# ── Yardımcı: Özet Mesajı ─────────────────────────────────────────────────────

def _build_recap_message(state: ConversationState) -> str:
    lines = ["📋 **Başvuru Özeti** — Bilgilerinizi kontrol edin:\n"]

    if state.get("vehicle_type") == "new":
        lines.append(f"• Araç Türü: Yeni Araç")
        if state.get("invoice_amount"):
            lines.append(f"• Proforma Fatura: {_fmt(state['invoice_amount'])} TL")
        if state.get("vehicle_model"):
            lines.append(f"• Araç Modeli: {state['vehicle_model']}")
        if state.get("requested_amount_new"):
            lines.append(f"• İstenen Finansman: {_fmt(state['requested_amount_new'])} TL")
        if state.get("guarantor_tckn"):
            from tools.tckn_tool import _mask
            lines.append(f"• Kefil TCKN: {_mask(state['guarantor_tckn'])}")
    else:
        lines.append(f"• Araç Türü: 2. El Araç")
        if state.get("kasko_value"):
            lines.append(f"• Kasko Değeri: {_fmt(state['kasko_value'])} TL")
        if state.get("vehicle_age"):
            lines.append(f"• Araç Yaşı: {state['vehicle_age']} yıl")
        if state.get("requested_amount_used"):
            lines.append(f"• İstenen Finansman: {_fmt(state['requested_amount_used'])} TL")
        if state.get("seller_tckn"):
            from tools.tckn_tool import _mask
            lines.append(f"• Satıcı TCKN: {_mask(state['seller_tckn'])}")

    lines.append("\n✅ Bilgiler doğruysa **'onaylıyorum'** yazın.")
    lines.append("✏️ Değişiklik için hangi alanı düzeltmek istediğinizi belirtin.")

    return "\n".join(lines)


def _fmt(amount: float) -> str:
    return f"{int(amount):,}".replace(",", ".")
