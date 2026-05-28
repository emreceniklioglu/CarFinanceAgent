"""
agents/intake_new.py — Yeni Araç Slot Toplama Agent'ı

AGENTIC PATTERN: Slot Filling + LLM Entity Extraction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hangi slot'un eksik olduğunu state'e bakarak bulur,
o slot için soru sorar veya müşterinin cevabından değeri çıkarır.

LLM'in görevi SADECE:
  - Sayı normalize (iki buçuk milyon → 2500000)
  - Entity extract (araç modeli: "bmw üç serisi" → BMW 3 Serisi)
  - TCKN ham metin çıkarımı

İş kuralı kararları (ticari mi, limit aşıldı mı) → validation_agent.
"""
import time
from langchain_core.messages import AIMessage
from llm.factory import get_router_llm
from llm.json_utils import parse_llm_json
from llm.prompts import SLOT_EXTRACTION_SYSTEM, VEHICLE_MODEL_EXTRACTION_SYSTEM
from graph.state import ConversationState
from observability.audit_logger import log_llm_call, log_slot_change
from tools.tckn_tool import _mask


# Sıralı slot listesi (hangi alan önce sorulacak)
NEW_CAR_SLOTS_ORDER = [
    "invoice_amount",
    "vehicle_model",
    "requested_amount_new",
    "guarantor_tckn",  # sadece kefil_required=True ise
]

SLOT_QUESTIONS = {
    "invoice_amount": "Aracın proforma fatura tutarı nedir? (TL olarak)",
    "vehicle_model": "Hangi marka ve model aracı düşünüyorsunuz?",
    "requested_amount_new": "Ne kadar finansman almak istiyorsunuz? (TL olarak)",
    "guarantor_tckn": "Proforma tutarı 5.000.000 TL üzerinde olduğu için kefil göstermeniz gerekmektedir. Kefilinizin TC kimlik numarası nedir?",
}


def get_next_missing_slot(state: ConversationState) -> str | None:
    """Sıradaki eksik slot'u döndürür; tümü doluysa None."""
    for slot in NEW_CAR_SLOTS_ORDER:
        if slot == "guarantor_tckn" and not state.get("kefil_required"):
            continue
        if not state.get(slot):
            return slot
    return None


def run_intake_new(state: ConversationState) -> dict:
    """
    Yeni araç intake node fonksiyonu.

    İki mod:
    A) Eksik slot var → soru sor
    B) Son mesaj bir cevap → parse et, slot'a yaz, doğrula
    """
    last_message = state["messages"][-1]
    user_text = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Eğer son mesaj AI ise (soru sorduk) → henüz cevap yok, bekle
    if hasattr(last_message, "type") and last_message.type == "ai":
        # Tüm slotlar zaten dolmuşsa intake_new_complete'i koru, recap akışı bozulmasın
        if get_next_missing_slot(state) is None:
            return {"current_step": "intake_new_complete", "awaited_slot": None}
        return {"current_step": "intake_new_waiting"}

    # Mevcut state'den ne beklediğimizi bul (awaited_slot alanından)
    awaited_slot = state.get("awaited_slot")
    import sys
    print(f"[DEBUG intake_new] awaited_slot={awaited_slot} user_text={user_text!r} current_step={state.get('current_step')} invoice={state.get('invoice_amount')} vtype={state.get('vehicle_type')}", flush=True)
    sys.stdout.flush()

    # Parse fazından gelen güncellemeleri biriktir, ama erken return etme;
    # parse başarılıysa aynı tur içinde sıradaki slotu da sormak istiyoruz.
    parse_updates: dict = {}
    if awaited_slot and awaited_slot in NEW_CAR_SLOTS_ORDER:
        updated = _parse_slot(state, awaited_slot, user_text)
        print(f"[DEBUG intake_new] parse result for slot '{awaited_slot}': {updated}", flush=True)
        if updated:
            parse_succeeded = awaited_slot in updated and updated[awaited_slot] is not None
            if parse_succeeded:
                updated["awaited_slot"] = None
                updated["edit_target_slot"] = None  # slot toplandı, edit hedefini temizle
                # flow_ended set edildiyse (ör. fatura üst limit aşımı) hemen dur
                if updated.get("flow_ended"):
                    return updated
                parse_updates = updated
                # Merge'i sıradaki-slot lookup'ı doğru görebilsin diye state üstüne uygula
                state = {**state, **parse_updates}
            else:
                # Parse başarısız → tekrar sor (zaten updated içinde messages var)
                return updated

    # Bir sonraki eksik slot'u bul ve sor
    next_slot = get_next_missing_slot(state)
    print(f"[DEBUG intake_new] next missing slot: {next_slot}", flush=True)
    if next_slot:
        question = SLOT_QUESTIONS[next_slot]
        return {
            **parse_updates,
            "current_step": f"asking_{next_slot}",
            "awaited_slot": next_slot,
            "messages": state["messages"] + [AIMessage(content=question)],
        }

    # Tüm slot'lar dolu → recap'e geç
    print(f"[DEBUG intake_new] all slots filled, proceeding to recap", flush=True)
    return {**parse_updates, "current_step": "intake_new_complete", "awaited_slot": None}


def _parse_slot(state: ConversationState, slot: str, user_text: str) -> dict | None:
    """LLM ile slot değerini parse et."""
    llm = get_router_llm()

    if slot in ("invoice_amount", "requested_amount_new"):
        return _parse_number_slot(state, slot, user_text, llm)
    elif slot == "vehicle_model":
        return _parse_vehicle_model(state, user_text, llm)
    elif slot == "guarantor_tckn":
        return _parse_tckn_slot(state, slot, user_text, llm)
    return None


def _parse_number_slot(state, slot, user_text, llm) -> dict:
    start = time.time()
    response = llm.invoke([
        {"role": "system", "content": SLOT_EXTRACTION_SYSTEM},
        {"role": "user", "content": user_text},
    ])
    import sys
    print(f"[DEBUG _parse_number_slot] slot={slot} user_text={user_text!r} llm_response={response.content!r}", flush=True)
    sys.stdout.flush()
    latency = int((time.time() - start) * 1000)
    log_llm_call(
        state["session_id"], "intake_new", llm.model, latency,
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
                AIMessage(content=f"Devam edelim — {question} (lütfen rakamla yazın, örn: 3000000 veya 3M TL)")
            ],
        }

    old_val = state.get(slot)
    log_slot_change(state["session_id"], slot, old_val, value)

    # Fatura üst limit hızlı kontrol — kapatma, yeniden sor
    if slot == "invoice_amount" and value > 7_000_000:
        msg = (f"Proforma fatura tutarı maksimum 7.000.000 TL olabilir "
               f"(girilen: {int(value):,} TL).\n"
               "Aracın proforma fatura tutarı nedir? (TL olarak)")
        return {
            "current_step": "asking_invoice_amount",
            "awaited_slot": "invoice_amount",
            "messages": state["messages"] + [AIMessage(content=msg)],
        }

    return {slot: value, "current_step": f"{slot}_collected"}


def _parse_vehicle_model(state, user_text, llm) -> dict:
    start = time.time()
    response = llm.invoke([
        {"role": "system", "content": VEHICLE_MODEL_EXTRACTION_SYSTEM},
        {"role": "user", "content": user_text},
    ])
    latency = int((time.time() - start) * 1000)
    log_llm_call(
        state["session_id"], "intake_new", llm.model, latency,
        len(user_text.split()), len(response.content.split()),
        prompt=[
            {"role": "system", "content": VEHICLE_MODEL_EXTRACTION_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        response=response.content,
    )

    parsed = parse_llm_json(response.content)
    if parsed:
        brand = parsed.get("brand", "")
        model = parsed.get("model", "")
        needs_clarification = parsed.get("needs_clarification", False)
    else:
        needs_clarification = True
        brand = model = ""

    if needs_clarification or not model:
        return {
            "current_step": "asking_vehicle_model",
            "awaited_slot": "vehicle_model",
            "messages": state["messages"] + [
                AIMessage(content="Marka ve modeli tam olarak belirtebilir misiniz? (örn: Toyota Corolla, BMW 3 Serisi)")
            ],
        }

    vehicle_model_str = f"{brand} {model}".strip()

    # Ticari araç kontrolü — kataloğa bakarak hemen filtrele
    from tools.catalog_tool import lookup_vehicle
    catalog_result = lookup_vehicle.invoke({"brand": brand, "model": model})
    if catalog_result["found"] and catalog_result["type"] == "ticari":
        msg = (f"Maalesef {vehicle_model_str} ticari bir araçtır. "
               "Bankamız yalnızca binek araçlara finansman sağlamaktadır. "
               "Farklı bir araç modeli girebilirsiniz.")
        return {
            "current_step": "asking_vehicle_model",
            "awaited_slot": "vehicle_model",
            "messages": state["messages"] + [AIMessage(content=msg)],
        }

    log_slot_change(state["session_id"], "vehicle_model", state.get("vehicle_model"), vehicle_model_str)
    return {"vehicle_model": vehicle_model_str, "current_step": "vehicle_model_collected"}


def _parse_tckn_slot(state, slot, user_text, llm) -> dict:
    # Skip intent
    skip_words = ["geç", "yok", "bilmiyorum", "pas", "atla", "istemiyorum"]
    if any(w in user_text.lower() for w in skip_words):
        if slot == "seller_tckn":
            log_slot_change(state["session_id"], slot, None, "skipped")
            return {slot: None, "current_step": f"{slot}_skipped"}

    import re
    # Ham TCKN veya input guard tarafından maskelenmiş (123****8910) her iki pattern'ı tanı
    match = re.search(r"\b[1-9]\d{10}\b", user_text)
    masked_match = re.search(r"\b([1-9]\d{2})\*{4}(\d{4})\b", user_text) if not match else None

    if match:
        tckn = match.group()
    elif masked_match:
        # Maskelenmiş geldi — ham hali yok, maskelenmiş hali sakla
        tckn = masked_match.group()
    else:
        return {
            "current_step": f"asking_{slot}",
            "awaited_slot": slot,
            "messages": state["messages"] + [
                AIMessage(content="TCKN'yi 11 haneli olarak tam yazabilir misiniz?")
            ],
        }

    masked = _mask(tckn) if match else tckn
    log_slot_change(state["session_id"], slot, None, masked)
    return {slot: tckn, "current_step": f"{slot}_collected"}
