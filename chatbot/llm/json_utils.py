"""LLM JSON cevaplarını güvenli parse etme yardımcıları.

LLM'ler bazen ```json ... ``` markdown code fence ile sarar veya
JSON öncesi/sonrası açıklama metni ekler. Bu helper bu varyasyonları temizler.
"""
import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def parse_llm_json(text: str) -> dict | None:
    """LLM cevabından JSON dict çıkar; başarısızsa None döner."""
    if not text:
        return None
    s = text.strip()
    # Markdown code fence'i kaldır
    s = _FENCE_RE.sub("", s).strip()
    # Direkt dene
    try:
        return json.loads(s)
    except Exception:
        pass
    # İlk { ... } bloğunu yakalamayı dene
    match = re.search(r"\{[\s\S]*\}", s)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None
    return None
