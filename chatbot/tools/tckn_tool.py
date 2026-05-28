"""
tools/tckn_tool.py — TCKN Doğrulama Aracı

AGENTIC PATTERN: Tool Use
━━━━━━━━━━━━━━━━━━━━━━━━
@tool decorator ile LangChain/LangGraph'a kayıt edilir.
Agent, "bu TCKN geçerli mi?" sorusunu cevaplamak için
bu tool'u çağırır. LLM kendi başına checksum hesaplamaz;
sadece hangi tool'u çağıracağına karar verir.

Algoritma: Türk TC Kimlik No mod-11 doğrulama standardı
- 11 haneli, ilk hane 0 olamaz
- 10. hane: (d1+d3+d5+d7+d9)*7 - (d2+d4+d6+d8) mod 10
- 11. hane: (d1+d2+...+d10) mod 10

API çağrısı YOK — tamamen deterministik, senkron.
"""
from langchain_core.tools import tool


@tool
def validate_tckn(tckn: str) -> dict:
    """
    Türk TC Kimlik Numarasını mod-11 algoritmasıyla doğrular.

    Args:
        tckn: Doğrulanacak TCKN (11 hane, string)

    Returns:
        {"valid": bool, "error": str | None, "masked": str}
    """
    # Boşluk ve tire temizle
    tckn = tckn.strip().replace(" ", "").replace("-", "")

    # Format kontrolü
    if not tckn.isdigit():
        return {"valid": False, "error": "TCKN yalnızca rakamlardan oluşmalı.", "masked": ""}
    if len(tckn) != 11:
        return {"valid": False, "error": f"TCKN 11 hane olmalı, {len(tckn)} hane girildi.", "masked": ""}
    if tckn[0] == "0":
        return {"valid": False, "error": "TCKN 0 ile başlayamaz.", "masked": ""}

    digits = [int(c) for c in tckn]

    # 10. hane kontrolü
    check10 = ((sum(digits[i] for i in range(0, 9, 2)) * 7) - sum(digits[i] for i in range(1, 8, 2))) % 10
    if digits[9] != check10:
        return {"valid": False, "error": "TCKN doğrulama hatası (10. hane).", "masked": _mask(tckn)}

    # 11. hane kontrolü
    check11 = sum(digits[:10]) % 10
    if digits[10] != check11:
        return {"valid": False, "error": "TCKN doğrulama hatası (11. hane).", "masked": _mask(tckn)}

    return {"valid": True, "error": None, "masked": _mask(tckn)}


def _mask(tckn: str) -> str:
    """TCKN'nin ortasını gizler: 123****8901 formatında döner."""
    if len(tckn) != 11:
        return "***"
    return tckn[:3] + "****" + tckn[7:]
