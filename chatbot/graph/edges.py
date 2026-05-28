"""
graph/edges.py — Koşullu Kenar Fonksiyonları

AGENTIC PATTERN: State Machine — Conditional Transitions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangGraph'ta conditional_edge, bir node'dan çıkan "karar noktalarını"
temsil eder. Her edge fonksiyonu state'i okur ve bir sonraki
node'un adını string olarak döndürür.

Bu dosya SADECE yönlendirme kararlarını içerir — iş mantığı yok.
İş mantığı: agent dosyalarında (intake, validation vb.)
"""
from graph.state import ConversationState
from observability.audit_logger import log_routing


def route_after_supervisor(state: ConversationState) -> str:
    """
    Supervisor'ın intent kararına göre hangi dala gidileceğini belirler.

    new_car  → intake_new node'una
    used_car → intake_used node'una
    faq      → faq_agent node'una
    out_of_scope / clarifying → END (akış kapanır veya tekrar bekler)
    """
    step = state.get("current_step", "")
    decision, reason = _decide_after_supervisor(state, step)

    try:
        sid = state.get("session_id")
        if sid:
            log_routing(sid, "supervisor", decision, reason)
    except Exception:
        pass  # routing log hatası akışı bozmasın

    return decision


def _decide_after_supervisor(state: ConversationState, step: str) -> tuple[str, str]:
    if state.get("flow_ended"):
        return "END", "flow_ended"

    # Başvuru zaten oluşturuldu → HGS adımındayız; recap/submission tekrarlanmamalı
    if state.get("application_id"):
        return "crosssell", "application already submitted, resume HGS step"

    if step == "clarifying":
        return "wait_for_input", "supervisor asked for clarification"

    intent = step.replace("supervisor_routed_", "")

    if intent == "faq":
        return "faq_agent", "faq intent"

    if intent == "new_car" or state.get("vehicle_type") == "new":
        # Bekleyen bir slot varsa önce kullanıcının cevabını parse et (veri kaybını önler)
        if state.get("awaited_slot"):
            return "intake_new", "parse awaited slot first"
        if _all_new_slots_filled(state):
            return "reflection", "new_car: all slots filled"
        return "intake_new", f"new_car intent (step={step})"

    if intent == "used_car" or state.get("vehicle_type") == "used":
        if state.get("awaited_slot"):
            return "intake_used", "parse awaited slot first"
        if _all_used_slots_filled(state):
            return "reflection", "used_car: all slots filled"
        return "intake_used", f"used_car intent (step={step})"

    return "END", f"no matching route (intent={intent!r}, step={step!r})"


def route_after_intake_new(state: ConversationState) -> str:
    """Yeni araç intake'i bitince nereye gideceğini belirler."""
    if state.get("flow_ended"):
        return "END"
    if state.get("current_step") == "intake_new_complete":
        return "validation_new"
    # Slot dolmadı → bir sonraki turda supervisor yeniden yönlendirir
    return "wait_for_input"


def route_after_intake_used(state: ConversationState) -> str:
    """2. El intake'i bitince nereye gideceğini belirler."""
    if state.get("flow_ended"):
        return "END"
    if state.get("current_step") == "intake_used_complete":
        return "validation_used"
    return "wait_for_input"


def route_after_validation(state: ConversationState) -> str:
    """Validation sonucuna göre yönlendir."""
    if state.get("flow_ended"):
        return "END"
    step = state.get("current_step", "")
    if step == "validation_passed":
        return "reflection"
    if step == "validation_needs_retry":
        # Hangi dal?
        return "intake_new" if state.get("vehicle_type") == "new" else "intake_used"
    return "END"


def route_after_reflection(state: ConversationState) -> str:
    """Reflection tamamsa recap'e, sorun varsa ilgili intake'e dön."""
    if state.get("flow_ended"):
        return "END"
    step = state.get("current_step", "")
    if step == "reflection_passed":
        return "recap"
    # Eksik slot varsa o intake'e geri dön
    return "intake_new" if state.get("vehicle_type") == "new" else "intake_used"


def route_after_recap(state: ConversationState) -> str:
    """Onay ekranı kararına göre yönlendir."""
    if state.get("recap_confirmed"):
        return "submission"
    if state.get("edit_target_slot"):
        # Hangi dal?
        return "intake_new" if state.get("vehicle_type") == "new" else "intake_used"
    # Cevap bekliyoruz
    return "wait_for_input"


def route_after_submission(state: ConversationState) -> str:
    if state.get("flow_ended") or state.get("current_step") == "submission_failed":
        return "END"
    return "crosssell"


def route_after_crosssell(state: ConversationState) -> str:
    if state.get("flow_ended"):
        return "END"
    if state.get("current_step") == "crosssell_waiting":
        return "wait_for_input"
    return "END"


def route_after_faq(state: ConversationState) -> str:
    """FAQ cevabından sonra turu bitir — kullanıcıdan yeni mesaj beklenir.

    NOT: supervisor'a geri DÖNMEMELİ. Aksi halde supervisor, FAQ agent'ın
    ürettiği AI cevabını yeni bir kullanıcı mesajı sanıp yeniden sınıflandırır
    ve bir döngüye girer (faq → supervisor → faq → ... → clarifying).
    Mid-flow resume bir sonraki turda supervisor'ın mid-flow bypass'ı ile olur.
    """
    return "wait_for_input"


# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────────

def _all_new_slots_filled(state: ConversationState) -> bool:
    base = all([state.get("invoice_amount"), state.get("vehicle_model"), state.get("requested_amount_new")])
    if not base:
        return False
    if state.get("kefil_required") and not state.get("guarantor_tckn"):
        return False
    return True


def _all_used_slots_filled(state: ConversationState) -> bool:
    return (
        bool(state.get("kasko_value"))
        and state.get("vehicle_age") is not None
        and bool(state.get("requested_amount_used"))
    )
