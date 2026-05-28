"""
agents/intake_used.py — 2. El Araç Slot Toplama Agent'ı

AGENTIC PATTERN: Slot Filling (State Machine'e bağlı)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
intake_new.py ile aynı pattern; 2. el dalının slot'ları farklı.
Satıcı TCKN opsiyoneldir — müşteri "bilmiyorum" diyebilir.
"""
import re
import time
from langchain_core.messages import AIMessage
from llm.factory import get_router_llm
from llm.json_utils import parse_llm_json
from llm.prompts import SLOT_EXTRACTION_SYSTEM
from graph.state import ConversationState
from observability.audit_logger import log_llm_call, log_slot_change
from tools.tckn_tool import _mask

USED_CAR_SLOTS_ORDER = ["kasko_value", "vehicle_age", "requested_amount_used", "seller_tckn"]

SLOT_QUESTIONS = {
    "kasko_value": "Aracın güncel kasko değeri nedir? (TL olarak)",
    "vehicle_age": "Aracın model yılı veya yaşı nedir? (örn: 2021 model veya 3 yaşında)",
    "requested_amount_used": "Ne kadar finansman almak istiyorsunuz? (TL olarak)",
    "seller_tckn": "Satıcının TC kimlik numarası nedir? (Bilmiyorsanız geçebilirsiniz)",
}


def get_next_missing_slot_used(state: ConversationState) -> str | None:
    for slot in USED_CAR_SLOTS_ORDER:
        val = state.get(slot)
        # "" → skipped (geçildi), None → henüz sorulmadı
        if val is None:
            return slot
    return None


def run_intake_used(state: ConversationState) -> dict:
    last_message = state["messages"][-1]
    user_text = last_message.content if hasattr(last_message, "content") else str(last_message)

    if hasattr(last_message, "type") and last_message.type == "ai":
        if get_next_missing_slot_used(state) is None:
            return {"current_step": "intake_used_complete", "awaited_slot": None}
        return {"current_step": "intake_used_waiting"}

    awaited_slot = state.get("awaited_slot")

    parse_updates: dict = {}
    if awaited_slot and awaited_slot in USED_CAR_SLOTS_ORDER:
        updated = _parse_slot_used(state, awaited_slot, user_text)
        if updated:
            parse_succeeded = (
                (awaited_slot in updated and updated[awaited_slot] is not None)
                or updated.get("current_step", "").endswith("_skipped")
            )
            if parse_succeeded:
                updated["awaited_slot"] = None
                updated["edit_target_slot"] = None  # slot toplandı, edit hedefini temizle
                if updated.get("flow_ended"):
                    return updated
                parse_updates = updated
                state = {**state, **parse_updates}
            else:
                # Parse başarısız → tekrar sor
                return updated

    next_slot = get_next_missing_slot_used(state)
    if next_slot:
        question = SLOT_QUESTIONS[next_slot]
        return {
            **parse_updates,
            "current_step": f"asking_{next_slot}",
            "awaited_slot": next_slot,
            "messages": state["messages"] + [AIMessage(content=question)],
        }

    return {**parse_updates, "current_step": "intake_used_complete", "awaited_slot": None}


def _parse_slot_used(state, slot, user_text) -> dict | None:
    llm = get_router_llm()

    if slot == "kasko_value":
        return _parse_number(state, slot, user_text, llm)

    elif slot == "vehicle_age":
        return _parse_age(state, user_text, llm)

    elif slot == "requested_amount_used":
        return _parse_number(state, slot, user_text, llm)

    elif slot == "seller_tckn":
        return _parse_optional_tckn(state, user_text)

    return None


def _parse_number(state, slot, user_text, llm) -> dict:
    start = time.time()
    response = llm.invoke([
        {"role": "system", "content": SLOT_EXTRACTION_SYSTEM},
        {"role": "user", "content": user_text},
    ])
    latency = int((time.time() - start) * 1000)
    log_llm_call(
        state["session_id"], "intake_used", llm.model, latency,
        len(user_text.split()), len(response.content.split()),
        prompt=[
            {"role": "system", "content": SLOT_EXTRACTION_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        response=response.content,
    )
    parsed = parse_llm_json(response.content)
    value = parsed.get("value") if parsed else None

    if value is None:
        question = SLOT_QUESTIONS.get(slot, "Lütfen rakamla yazın.")
        return {
            "current_step": f"asking_{slot}",
            "awaited_slot": slot,
            "messages": state["messages"] + [
                AIMessage(content=f"Devam edelim — {question} (lütfen rakamla yazın, örn: 500000 veya 500bin TL)")
            ],
        }
    log_slot_change(state["session_id"], slot, state.get(slot), value)
    return {slot: value, "current_step": f"{slot}_collected"}


def _parse_age(state, user_text, llm) -> dict:
    """Model yılı veya yaş ifadesini parse eder."""
    import datetime
    current_year = datetime.date.today().year

    # Önce direkt yıl ara: 2020, 2021 vb.
    year_match = re.search(r"\b(20\d{2})\b", user_text)
    if year_match:
        age = current_year - int(year_match.group(1))
        log_slot_change(state["session_id"], "vehicle_age", state.get("vehicle_age"), age)

        if age < 0:
            msg = (f"{year_match.group(1)} henüz geçmiş bir model yılı değil. "
                   "Lütfen geçerli bir model yılı giriniz (örn: 2021, 2022).")
            return {
                "current_step": "asking_vehicle_age",
                "awaited_slot": "vehicle_age",
                "messages": state["messages"] + [AIMessage(content=msg)],
            }
        if age > 5:
            msg = (f"Araç yaşı {age} yıl olarak hesaplandı; maksimum kabul edilen 5 yıldır. "
                   "Lütfen 5 yıl veya daha küçük bir model yılı/yaş giriniz.")
            return {
                "current_step": "asking_vehicle_age",
                "awaited_slot": "vehicle_age",
                "messages": state["messages"] + [AIMessage(content=msg)],
            }
        return {"vehicle_age": age, "current_step": "vehicle_age_collected"}

    # LLM ile parse et
    start = time.time()
    response = llm.invoke([
        {"role": "system", "content": SLOT_EXTRACTION_SYSTEM},
        {"role": "user", "content": f"Araç yaşı: {user_text}"},
    ])
    latency = int((time.time() - start) * 1000)
    log_llm_call(
        state["session_id"], "intake_used", llm.model, latency, 10, 10,
        prompt=[
            {"role": "system", "content": SLOT_EXTRACTION_SYSTEM},
            {"role": "user", "content": f"Araç yaşı: {user_text}"},
        ],
        response=response.content,
    )

    parsed = parse_llm_json(response.content)
    try:
        age = int(parsed.get("value", 0)) if parsed else None
    except Exception:
        age = None

    if age is None:
        return {
            "current_step": "asking_vehicle_age",
            "awaited_slot": "vehicle_age",
            "messages": state["messages"] + [
                AIMessage(content="Yaşı anlayamadım. Model yılını (örn: 2021) veya kaç yaşında olduğunu yazar mısınız?")
            ],
        }

    if age > 5:
        msg = (f"Araç yaşı {age} yıl olarak alındı; maksimum kabul edilen 5 yıldır. "
               "Lütfen 5 yıl veya daha küçük bir model yılı/yaş giriniz.")
        return {
            "current_step": "asking_vehicle_age",
            "awaited_slot": "vehicle_age",
            "messages": state["messages"] + [AIMessage(content=msg)],
        }

    log_slot_change(state["session_id"], "vehicle_age", state.get("vehicle_age"), age)
    return {"vehicle_age": age, "current_step": "vehicle_age_collected"}


def _parse_optional_tckn(state, user_text) -> dict:
    skip_words = ["geç", "yok", "bilmiyorum", "pas", "atla", "istemiyorum", "hayır"]
    if any(w in user_text.lower() for w in skip_words):
        return {"seller_tckn": "", "current_step": "seller_tckn_skipped"}

    # Ham TCKN veya input guard tarafından maskelenmiş (789****2355) pattern
    match = re.search(r"\b[1-9]\d{10}\b", user_text)
    masked_match = re.search(r"\b[1-9]\d{2}\*{4}\d{4}\b", user_text) if not match else None

    if match:
        tckn = match.group()
        masked = _mask(tckn)
    elif masked_match:
        tckn = masked_match.group()
        masked = tckn  # zaten maskeli
    else:
        return {
            "current_step": "asking_seller_tckn",
            "awaited_slot": "seller_tckn",
            "messages": state["messages"] + [
                AIMessage(content="TCKN 11 haneli olmalıdır. Bilmiyorsanız 'geç' yazabilirsiniz.")
            ],
        }

    log_slot_change(state["session_id"], "seller_tckn", None, masked)
    return {"seller_tckn": tckn, "current_step": "seller_tckn_collected"}
