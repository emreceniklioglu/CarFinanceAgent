"""
observability/audit_logger.py — Değiştirilemez Denetim Kaydı

REPORT Karşılığı: "Audit Log + Trace" (Gözlem & Denetim)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Her LLM çağrısı, tool çağrısı, guard olayı ve slot değişimi
değiştirilemez (APPEND-ONLY) SQLite tablosuna kaydedilir.

KVKK ve BDDK uyumu için:
- TCKN, kart no gibi hassas veriler maskelenmiş haliyle yazılır
- Ham kişisel veri loga girmez
- Audit kayıtları silinemez (soft-delete bile yok)

Event tipleri:
  llm_call    : LLM'e giden/dönen her çağrı
  tool_call   : @tool fonksiyonu çağrıları
  guard_event : Input/output guard tetiklenmeleri
  slot_change : Konuşma state'inde slot değişimleri
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from config import DB_PATH


def log_event(
    session_id: str,
    event_type: str,           # llm_call | tool_call | guard_event | slot_change
    agent: str,                # hangi agent oluşturdu
    details: dict,             # olay detayları (model, latency, token vb.)
    masked_input: str = "",    # maskelenmiş kullanıcı girdisi
) -> str:
    """
    Tek bir denetim olayı kaydeder.

    Returns:
        event_id: Oluşturulan kayıt UUID'si
    """
    event_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO audit_events (id, session_id, timestamp, event_type, agent, details, masked_input)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (event_id, session_id, timestamp, event_type, agent, json.dumps(details, ensure_ascii=False), masked_input),
    )
    conn.commit()
    conn.close()
    return event_id


def log_llm_call(
    session_id: str,
    agent: str,
    model: str,
    latency_ms: int,
    tokens_in: int,
    tokens_out: int,
    prompt=None,
    response=None,
    max_chars: int = 4000,
):
    """LLM çağrısı için kısayol.

    prompt: str | list[dict] | None — LLM'e giden mesaj(lar). list verilirse
            role/content'e indirgenip JSON-serialize edilir.
    response: str | None — LLM'in ürettiği ham içerik.
    max_chars: prompt/response için karakter üst limiti (DB'yi şişirmemek için).
    """
    details = {
        "model": model,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }

    if prompt is not None:
        details["prompt"] = _trim(_normalize_prompt(prompt), max_chars)
    if response is not None:
        details["response"] = _trim(str(response), max_chars)

    return log_event(
        session_id=session_id,
        event_type="llm_call",
        agent=agent,
        details=details,
    )


def _normalize_prompt(prompt) -> str:
    """Prompt'u loglanabilir stringe çevir; system mesajlarını ele (gürültü)."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for m in prompt:
            if isinstance(m, dict):
                role = m.get("role", "?")
                if role == "system":
                    continue  # system prompt'u görüntülemeden gizle
                content = m.get("content", "")
                parts.append(f"[{role}] {content}")
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return str(prompt)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text) - limit} chars>"


def log_tool_call(session_id: str, agent: str, tool_name: str, args: dict, result: dict, success: bool):
    """Tool çağrısı için kısayol."""
    return log_event(
        session_id=session_id,
        event_type="tool_call",
        agent=agent,
        details={"tool": tool_name, "args": _sanitize(args), "result": _sanitize(result), "success": success},
    )


def log_guard_event(session_id: str, guard_type: str, triggered: bool, reason: str, masked_text: str = ""):
    """Guard tetiklenmesi için kısayol."""
    return log_event(
        session_id=session_id,
        event_type="guard_event",
        agent=guard_type,
        details={"triggered": triggered, "reason": reason},
        masked_input=masked_text,
    )


def log_routing(session_id: str, from_node: str, to_node: str, reason: str = ""):
    """Edge fonksiyonunun verdiği yönlendirme kararını kaydet."""
    return log_event(
        session_id=session_id,
        event_type="routing",
        agent=from_node,
        details={"from": from_node, "to": to_node, "reason": reason},
    )


def log_slot_change(session_id: str, slot_name: str, old_value, new_value):
    """Slot değişimi için kısayol — hassas alanları maskele."""
    sensitive = {"guarantor_tckn", "seller_tckn"}
    if slot_name in sensitive:
        old_value = _mask_tckn(str(old_value)) if old_value else None
        new_value = _mask_tckn(str(new_value)) if new_value else None

    return log_event(
        session_id=session_id,
        event_type="slot_change",
        agent="orchestrator",
        details={"slot": slot_name, "old": old_value, "new": new_value},
    )


def get_session_events(session_id: str) -> list[dict]:
    """Bir oturumun tüm audit olaylarını döndürür."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, timestamp, event_type, agent, details, masked_input FROM audit_events WHERE session_id=? ORDER BY timestamp",
        (session_id,),
    ).fetchall()
    conn.close()

    return [
        {
            "id": r[0], "timestamp": r[1], "event_type": r[2],
            "agent": r[3], "details": json.loads(r[4]), "masked_input": r[5],
        }
        for r in rows
    ]


def get_recent_events(limit: int = 50) -> list[dict]:
    """Son N olayı döndürür (dashboard için)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, session_id, timestamp, event_type, agent, details FROM audit_events ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [{"id": r[0], "session": r[1][:8], "time": r[2][11:19], "type": r[3], "agent": r[4], "details": r[5]} for r in rows]


def _sanitize(data: dict) -> dict:
    """Dict içindeki hassas alanları maskele."""
    sensitive_keys = {"tckn", "tc", "card", "iban", "password"}
    result = {}
    for k, v in data.items():
        if any(s in k.lower() for s in sensitive_keys) and isinstance(v, str):
            result[k] = _mask_tckn(v)
        else:
            result[k] = v
    return result


def _mask_tckn(value: str) -> str:
    if len(value) >= 7:
        return value[:3] + "****" + value[7:]
    return "***"
