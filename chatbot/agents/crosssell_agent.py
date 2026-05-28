"""
agents/crosssell_agent.py — HGS Çapraz Satış Agent'ı

AGENTIC PATTERN: Scoped Tool Access (Least-Privilege)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bu agent YALNIZCA HGS servisine erişebilir.
Ön Başvuru DB'ye dokunamaz — bu least-privilege prensibin uygulaması.

İki adım:
1. check_hgs_status → aktif HGS varsa adımı atla
2. Teklif sun → onaylanırsa register_hgs çağır
"""
from langchain_core.messages import AIMessage
from tools.hgs_tool import check_hgs_status, register_hgs
from graph.state import ConversationState
from observability.audit_logger import log_tool_call, log_slot_change

HGS_ACCEPT_WORDS = ["evet", "istiyorum", "tamam", "olur", "onaylıyorum"]
HGS_REJECT_WORDS = ["hayır", "istemiyorum", "geç", "atla", "vazgeçtim", "lazım değil", "hayir"]


def run_crosssell(state: ConversationState) -> dict:
    """HGS çapraz satış node fonksiyonu."""

    # Adım 1: Aktif HGS var mı?
    hgs_status = check_hgs_status.invoke({"session_id": state["session_id"]})
    log_tool_call(state["session_id"], "crosssell_agent", "check_hgs_status",
                  {}, hgs_status, True)

    if hgs_status["has_active_hgs"]:
        # HGS zaten var → adımı atla
        return {
            "current_step": "crosssell_skipped",
            "hgs_consent": False,
            "messages": state["messages"] + [
                AIMessage(content="HGS'iniz zaten aktif. İyi sürüşler! 🚗\n\nBaşvurunuz için teşekkürler.")
            ],
            "flow_ended": True,
        }

    # Adım 2: Müşterinin son mesajı HGS teklifine cevap mı?
    last_message = state["messages"][-1]
    user_text = last_message.content if hasattr(last_message, "content") else ""

    if hasattr(last_message, "type") and last_message.type == "ai":
        # Teklifi zaten gönderdik, cevap bekliyoruz
        return {"current_step": "crosssell_waiting"}

    lower = user_text.lower()

    if any(w in lower for w in HGS_REJECT_WORDS):
        log_slot_change(state["session_id"], "hgs_consent", None, False)
        return {
            "current_step": "crosssell_declined",
            "hgs_consent": False,
            "flow_ended": True,
            "messages": state["messages"] + [
                AIMessage(content="Anlaşıldı. Başvurunuz için teşekkürler, iyi günler! 🙂")
            ],
        }

    if any(w in lower for w in HGS_ACCEPT_WORDS):
        # HGS kaydı yap
        hgs_result = register_hgs.invoke({
            "session_id": state["session_id"],
            "application_id": state.get("application_id", ""),
        })
        log_tool_call(state["session_id"], "crosssell_agent", "register_hgs",
                      {"application_id": state.get("application_id")},
                      hgs_result, hgs_result["success"])
        log_slot_change(state["session_id"], "hgs_consent", None, True)

        if hgs_result["success"]:
            msg = f"HGS'iniz başarıyla tanımlandı! (HGS No: {hgs_result['hgs_id']})\n\nTeşekkürler, iyi sürüşler! 🚗"
        else:
            msg = "HGS tanımlamasında bir sorun oluştu. Şubemizden destek alabilirsiniz."

        return {
            "current_step": "crosssell_complete",
            "hgs_consent": True,
            "flow_ended": True,
            "messages": state["messages"] + [AIMessage(content=msg)],
        }

    # İlk kez geliyorsa teklif gösterildi mi? (submission_agent zaten HGS_PITCH'i gönderdi)
    return {"current_step": "crosssell_waiting"}
