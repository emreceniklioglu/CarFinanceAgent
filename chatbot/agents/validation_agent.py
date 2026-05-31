"""
agents/validation_agent.py — Doğrulama Agent'ı (Deterministik Tool Use)

AGENTIC PATTERN: Tool Use (Deterministik — LLM yok)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bu agent LLM ÇAĞIRMAZ; iş kuralı tool'larını kod ile sabit sırayla çağırır.
LLM güdümlü reason→act→observe (ReAct) döngüsü YOKTUR — akış deterministiktir.

Bu agent:
1. Mevcut state'deki slot değerlerini alır
2. İş kuralı tool'larını çağırarak doğrular
3. Hata bulursa müşteriye açıklayıcı mesaj gönderir
4. Tüm kurallar geçerse bir sonraki adıma onay verir

NEDEN deterministik?
- İş kuralları LLM'de değil, tool'larda → halüsinasyon riski 0
- Bankacılık doğrulaması denetlenebilir ve tekrarlanabilir olmalı
- Hangi kontrolün ne zaman yapılacağı koda gömülü, LLM'e bırakılmaz

Araç erişimi: validate_new_car_amount, validate_used_car_amount
"""
from langchain_core.messages import AIMessage
from tools.amount_tool import validate_new_car_amount, validate_used_car_amount
from graph.state import ConversationState
from observability.audit_logger import log_tool_call


def validate_new_car_state(state: ConversationState) -> dict:
    """
    Yeni araç dalındaki tüm slot'ları sırayla doğrular.
    İlk hata bulunduğunda müşteriye mesaj döner ve akış durur.
    """
    errors = []
    messages_to_add = []

    # Ticari araç kontrolü intake_new'da yapılıyor — validation'da tekrar kontrol gerekmez

    # ── Finansman Tutarı Kontrolü ─────────────────────────────────────────────
    if state.get("invoice_amount") and state.get("requested_amount_new"):
        result = validate_new_car_amount.invoke({
            "invoice_amount": state["invoice_amount"],
            "requested_amount": state["requested_amount_new"],
        })
        log_tool_call(state["session_id"], "validation_agent", "validate_new_car_amount",
                      {"invoice": state["invoice_amount"], "requested": state["requested_amount_new"]},
                      result, result["valid"])

        if not result["valid"]:
            msg = (f"{result['error']}\n"
                   "Ne kadar finansman almak istiyorsunuz? (TL olarak)")
            return {
                "current_step": "validation_needs_retry",
                "requested_amount_new": None,
                "awaited_slot": "requested_amount_new",
                "kefil_required": result.get("kefil_required", False),
                "errors": state["errors"] + [result["error"]],
                "messages": state["messages"] + [AIMessage(content=msg)],
                "flow_ended": False,
            }

        # Kefil zorunluluğunu state'e yaz
        if result.get("kefil_required"):
            return {
                "current_step": "validation_passed",
                "kefil_required": True,
                "errors": state["errors"],
                "messages": state["messages"],
                "flow_ended": False,
            }

    # ── Kefil TCKN Kontrolü ───────────────────────────────────────────────────
    # TCKN mod-11 doğrulaması demo'da devre dışı — sadece 11 hane kontrolü intake'te yapılıyor

    return {
        "current_step": "validation_passed",
        "errors": state["errors"],
        "messages": state["messages"],
        "flow_ended": False,
    }


def validate_used_car_state(state: ConversationState) -> dict:
    """2. El araç slot'larını doğrular."""
    errors = []
    messages_to_add = []

    if state.get("kasko_value") and state.get("vehicle_age") is not None and state.get("requested_amount_used"):
        result = validate_used_car_amount.invoke({
            "kasko_value": state["kasko_value"],
            "vehicle_age": state["vehicle_age"],
            "requested_amount": state["requested_amount_used"],
        })
        log_tool_call(state["session_id"], "validation_agent", "validate_used_car_amount",
                      {"kasko": state["kasko_value"], "age": state["vehicle_age"],
                       "requested": state["requested_amount_used"]},
                      result, result["valid"])

        if not result["valid"]:
            msg = (f"{result['error']} Maksimum: {_fmt(result['max_allowed'])} TL.\n"
                   "Ne kadar finansman almak istiyorsunuz? (TL olarak)")
            return {
                "current_step": "validation_needs_retry",
                "errors": state["errors"] + [result['error']],
                "messages": state["messages"] + [AIMessage(content=msg)],
                "requested_amount_used": None,
                "awaited_slot": "requested_amount_used",
                "flow_ended": False,
            }

    # Satıcı TCKN mod-11 doğrulaması demo'da devre dışı

    return {
        "current_step": "validation_passed",
        "errors": state["errors"],
        "messages": state["messages"],
        "flow_ended": False,
    }


def _fmt(amount: float) -> str:
    return f"{int(amount):,}".replace(",", ".")
