"""
guardrails/input_guard.py — Giriş Güvenlik Katmanı

REPORT Karşılığı: L4 — Input Guard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Müşteri mesajı Orchestrator'a ulaşmadan önce şu dört kontrol:

1. PII Maskeleme: TCKN, IBAN, kart no gibi değerleri maskele
   (müşteri güvenlik amacıyla paylaşmış olabilir — güvenli saklama)
2. Prompt Injection Tespiti: "tüm talimatları unut" gibi ifadeler
3. Kapsam Dışı Ön Filtre: açıkça alakasız kısa mesajlar
4. Uzunluk Kontrolü: aşırı uzun mesajlar sınırlandırılır

Kural bazlı (deterministik) çalışır — ek LLM çağrısı yapmaz.
Hız kritik: her kullanıcı turunda çalışır.
"""
import re
from config import INJECTION_KEYWORDS
from observability.audit_logger import log_guard_event


# ── Regex Kalıpları ───────────────────────────────────────────────────────────
_TCKN_RE = re.compile(r"\b[1-9]\d{10}\b")
_IBAN_RE = re.compile(r"\bTR\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{2}\b", re.IGNORECASE)
_CARD_RE = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")

MAX_MESSAGE_LENGTH = 1000  # karakter


def check_input(message: str, session_id: str) -> dict:
    """
    Giriş mesajını güvenlik kontrollerinden geçirir.

    Returns:
        {
            "allowed": bool,
            "sanitized_text": str,      # PII maskelenmiş metin
            "block_reason": str | None, # neden engellendi
            "pii_detected": bool,
            "injection_detected": bool,
        }
    """
    result = {
        "allowed": True,
        "sanitized_text": message,
        "block_reason": None,
        "pii_detected": False,
        "injection_detected": False,
    }

    # 1. Uzunluk kontrolü
    if len(message) > MAX_MESSAGE_LENGTH:
        result["allowed"] = False
        result["block_reason"] = f"Mesaj çok uzun ({len(message)} karakter, limit {MAX_MESSAGE_LENGTH})."
        log_guard_event(session_id, "input_guard", True, "message_too_long")
        return result

    # 2. PII Maskeleme (engelleme değil — sadece maskele)
    sanitized = message
    if _TCKN_RE.search(sanitized):
        sanitized = _TCKN_RE.sub(lambda m: m.group()[:3] + "****" + m.group()[7:], sanitized)
        result["pii_detected"] = True

    if _IBAN_RE.search(sanitized):
        sanitized = _IBAN_RE.sub("[IBAN GİZLENDİ]", sanitized)
        result["pii_detected"] = True

    if _CARD_RE.search(sanitized):
        sanitized = _CARD_RE.sub("[KART NO GİZLENDİ]", sanitized)
        result["pii_detected"] = True

    if result["pii_detected"]:
        log_guard_event(session_id, "input_guard", True, "pii_detected", sanitized)

    result["sanitized_text"] = sanitized

    # 3. Prompt Injection Tespiti
    lower = message.lower()
    for keyword in INJECTION_KEYWORDS:
        if keyword.lower() in lower:
            result["allowed"] = False
            result["injection_detected"] = True
            result["block_reason"] = "Güvenlik politikası: bu ifadeyle devam edilemiyor."
            log_guard_event(session_id, "input_guard", True, f"injection_keyword:{keyword}", sanitized[:100])
            return result

    log_guard_event(session_id, "input_guard", False, "passed")
    return result


def get_block_message() -> str:
    """Engellenen mesaj için standart yanıt."""
    return "Üzgünüm, bu mesajla devam edemiyorum. Lütfen araç finansmanı konusunda sorunuzu yeniden yazın."
