"""
guardrails/output_guard.py — Çıkış Güvenlik Katmanı

REPORT Karşılığı: L7 — Output Guard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLM çıktısı müşteriye iletilmeden önce üç kontrol:

1. PII Redaction: Çıktıda kalan hassas veri parçaları temizlenir
2. Finansal Taahhüt Filtresi: Faiz oranı garantisi, kesin onay vb.
   ifadeler engellenir → yasal ve itibar riski önlenir
3. FAQ Kaynak Zorunluluğu: FAQ cevabı referans chunk olmadan geçemez

Basit kural listesi tabanlı — ek LLM çağrısı yapmaz.
"""
import re
from observability.audit_logger import log_guard_event

# Yasak finansal taahhüt ifadeleri (Türkçe + İngilizce)
FINANCIAL_COMMITMENT_PATTERNS = [
    r"%\s*0[,.]?[05]?\s*(faiz|interest)",
    r"kesinlikle onaylan",
    r"garantili kredi",
    r"anında onay",
    r"100\s*%\s*onay",
    r"faiz\s*almıyoruz",
    r"faizsiz",
    r"guaranteed approval",
    r"interest free",
]

_TCKN_RE = re.compile(r"\b[1-9]\d{10}\b")
_IBAN_RE = re.compile(r"\bTR\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{2}\b", re.IGNORECASE)


def check_output(text: str, session_id: str, is_faq_response: bool = False, source_chunks: list = None) -> dict:
    """
    LLM çıktısını güvenlik kontrollerinden geçirir.

    Args:
        text: LLM'in ürettiği metin
        session_id: Audit log için
        is_faq_response: FAQ cevabıysa kaynak chunk kontrolü yapılır
        source_chunks: RAG'dan gelen kaynak chunk'lar

    Returns:
        {
            "allowed": bool,
            "cleaned_text": str,
            "block_reason": str | None,
            "redacted_pii": bool,
            "financial_block": bool,
        }
    """
    result = {
        "allowed": True,
        "cleaned_text": text,
        "block_reason": None,
        "redacted_pii": False,
        "financial_block": False,
    }

    # 1. PII Redaction
    cleaned = text
    if _TCKN_RE.search(cleaned):
        cleaned = _TCKN_RE.sub("[KİŞİSEL VERİ GİZLENDİ]", cleaned)
        result["redacted_pii"] = True
        log_guard_event(session_id, "output_guard", True, "pii_in_output")

    if _IBAN_RE.search(cleaned):
        cleaned = _IBAN_RE.sub("[IBAN GİZLENDİ]", cleaned)
        result["redacted_pii"] = True

    result["cleaned_text"] = cleaned

    # 2. Finansal Taahhüt Filtresi
    lower = text.lower()
    for pattern in FINANCIAL_COMMITMENT_PATTERNS:
        if re.search(pattern, lower):
            result["allowed"] = False
            result["financial_block"] = True
            result["block_reason"] = "Finansal taahhüt ifadesi tespit edildi."
            log_guard_event(session_id, "output_guard", True, f"financial_commitment:{pattern[:30]}")
            return result

    # 3. FAQ kaynak zorunluluğu
    if is_faq_response and not source_chunks:
        result["allowed"] = False
        result["block_reason"] = "FAQ cevabı kaynak chunk olmadan iletilemez."
        log_guard_event(session_id, "output_guard", True, "faq_no_source")
        return result

    log_guard_event(session_id, "output_guard", False, "passed")
    return result


def get_financial_block_message() -> str:
    return "Bu konuda kesin bilgi veremiyorum. Güncel faiz oranları ve koşullar için lütfen şubemizi veya müşteri hizmetlerimizi arayın."
