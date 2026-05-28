"""
tools/amount_tool.py — Finansman Tutarı Hesap ve Doğrulama

AGENTIC PATTERN: Tool Use — Deterministik İş Kuralları
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tüm finansman kuralları (üst limitler, oranlar) burada KOD olarak
çalışır. LLM bu kuralları ne hesaplar ne de doğrular — sadece
hangi tool'u çağıracağını seçer.

Bu sayede:
- Kural hataları test edilebilir (birim test)
- Kural değişikliği için tek bir yer
- LLM halüsinasyonu → iş kararlarına etki edemez
"""
from langchain_core.tools import tool
from config import (
    NEW_CAR_MAX_INVOICE,
    NEW_CAR_FINANCE_RATIO,
    NEW_CAR_GUARANTOR_THRESHOLD,
    USED_CAR_MAX_AGE,
    USED_CAR_FINANCE_RATIO,
    USED_CAR_MAX_FINANCE,
)


@tool
def validate_new_car_amount(invoice_amount: float, requested_amount: float) -> dict:
    """
    Yeni araç finansman tutarını iş kurallarına göre doğrular.

    Kurallar:
    - Fatura tutarı ≤ 7.000.000 TL
    - Finansman ≤ %60 × fatura tutarı

    Returns:
        {"valid": bool, "max_allowed": float, "error": str | None,
         "kefil_required": bool}
    """
    # Kural 1: Fatura üst limiti
    if invoice_amount > NEW_CAR_MAX_INVOICE:
        return {
            "valid": False,
            "max_allowed": NEW_CAR_MAX_INVOICE,
            "error": f"Proforma fatura tutarı {_fmt(NEW_CAR_MAX_INVOICE)} TL üst limitini aşıyor.",
            "kefil_required": False,
        }

    # Kural 2: Finansman oranı
    max_finance = invoice_amount * NEW_CAR_FINANCE_RATIO
    if requested_amount > max_finance:
        return {
            "valid": False,
            "max_allowed": max_finance,
            "error": (
                f"İstenen finansman ({_fmt(requested_amount)} TL), "
                f"fatura tutarının %{int(NEW_CAR_FINANCE_RATIO*100)}'ini aşıyor. "
                f"Maksimum: {_fmt(max_finance)} TL"
            ),
            "kefil_required": invoice_amount >= NEW_CAR_GUARANTOR_THRESHOLD,
        }

    return {
        "valid": True,
        "max_allowed": max_finance,
        "error": None,
        "kefil_required": invoice_amount >= NEW_CAR_GUARANTOR_THRESHOLD,
    }


@tool
def validate_used_car_amount(kasko_value: float, vehicle_age: int, requested_amount: float) -> dict:
    """
    2. El araç finansman tutarını iş kurallarına göre doğrular.

    Kurallar:
    - Araç yaşı ≤ 5 yıl
    - Finansman ≤ min(%40 × kasko_value, 3.000.000 TL)

    Returns:
        {"valid": bool, "max_allowed": float, "error": str | None}
    """
    # Kural 1: Yaş kontrolü
    if vehicle_age > USED_CAR_MAX_AGE:
        return {
            "valid": False,
            "max_allowed": 0,
            "error": f"Araç yaşı {vehicle_age} yıl, maksimum kabul edilen {USED_CAR_MAX_AGE} yıldır.",
        }

    # Kural 2: Finansman oranı (min ile sınırla)
    ratio_limit = kasko_value * USED_CAR_FINANCE_RATIO
    max_finance = min(ratio_limit, USED_CAR_MAX_FINANCE)

    if requested_amount > max_finance:
        return {
            "valid": False,
            "max_allowed": max_finance,
            "error": (
                f"İstenen finansman ({_fmt(requested_amount)} TL) limiti aşıyor. "
                f"Kasko oranı limiti: {_fmt(ratio_limit)} TL, "
                f"mutlak üst limit: {_fmt(USED_CAR_MAX_FINANCE)} TL."
            ),
        }

    return {"valid": True, "max_allowed": max_finance, "error": None}


@tool
def calculate_max_finance_new(invoice_amount: float) -> dict:
    """Yeni araç için alınabilecek maksimum finansmanı hesaplar."""
    max_f = invoice_amount * NEW_CAR_FINANCE_RATIO
    return {
        "max_finance": max_f,
        "ratio": NEW_CAR_FINANCE_RATIO,
        "kefil_required": invoice_amount >= NEW_CAR_GUARANTOR_THRESHOLD,
    }


@tool
def calculate_max_finance_used(kasko_value: float) -> dict:
    """2. El araç için alınabilecek maksimum finansmanı hesaplar."""
    ratio_limit = kasko_value * USED_CAR_FINANCE_RATIO
    max_f = min(ratio_limit, USED_CAR_MAX_FINANCE)
    return {"max_finance": max_f, "ratio": USED_CAR_FINANCE_RATIO, "absolute_cap": USED_CAR_MAX_FINANCE}


def _fmt(amount: float) -> str:
    """Sayıyı Türkçe binlik ayırıcıyla formatlar: 3000000 → '3.000.000'"""
    return f"{int(amount):,}".replace(",", ".")
