"""
agents/submission_agent.py — Başvuru Kayıt Agent'ı

AGENTIC PATTERN: Tool Use (write_application)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Müşteri tüm bilgileri onayladıktan sonra bu agent çalışır.
db_tool aracılığıyla SQLite'a yazar; LLM doğrudan DB'ye erişemez.

Adımlar:
1. State'deki tüm slot'ları topla
2. TCKN maskelenmiş haliyle gönder
3. write_application tool'unu çağır
4. UUID başvuru numarasını state'e yaz
5. Müşteriye tebrik mesajı + HGS geçişine hazırlan
"""
from langchain_core.messages import AIMessage
from tools.db_tool import write_application
from tools.tckn_tool import _mask
from graph.state import ConversationState
from observability.audit_logger import log_tool_call
from llm.prompts import HGS_PITCH


def run_submission(state: ConversationState) -> dict:
    """Onaylanmış başvuruyu veritabanına yazar."""

    vehicle_type = state.get("vehicle_type")

    # TCKN'leri maskele — ham değer DB'ye girmez
    guarantor_masked = _mask(state["guarantor_tckn"]) if state.get("guarantor_tckn") else None
    seller_masked = _mask(state["seller_tckn"]) if state.get("seller_tckn") else None

    result = write_application.invoke({
        "session_id": state["session_id"],
        "vehicle_type": vehicle_type,
        "invoice_amount": state.get("invoice_amount"),
        "vehicle_model": state.get("vehicle_model"),
        "requested_amount_new": state.get("requested_amount_new"),
        "guarantor_tckn_masked": guarantor_masked,
        "kasko_value": state.get("kasko_value"),
        "vehicle_age": state.get("vehicle_age"),
        "requested_amount_used": state.get("requested_amount_used"),
        "seller_tckn_masked": seller_masked,
    })

    log_tool_call(
        session_id=state["session_id"],
        agent="submission_agent",
        tool_name="write_application",
        args={"vehicle_type": vehicle_type},
        result={"success": result["success"], "application_id": result.get("application_id")},
        success=result["success"],
    )

    if not result["success"]:
        error_msg = f"Başvuru kaydedilirken bir hata oluştu: {result['error']}. Lütfen tekrar deneyin."
        return {
            "current_step": "submission_failed",
            "errors": state["errors"] + [error_msg],
            "messages": state["messages"] + [AIMessage(content=error_msg)],
        }

    app_id = result["application_id"]
    hgs_pitch = HGS_PITCH.format(application_id=app_id)

    return {
        "current_step": "submission_complete",
        "application_id": app_id,
        "messages": state["messages"] + [AIMessage(content=hgs_pitch)],
    }
