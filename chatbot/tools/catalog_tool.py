"""
tools/catalog_tool.py — Araç Modeli Kataloğu

AGENTIC PATTERN: Tool Use
━━━━━━━━━━━━━━━━━━━━━━━━
Araç modeli → binek/ticari sınıflandırması bu tool ile yapılır.
LLM modeli çıkarır, katalog nihai kararı verir.
O(1) lookup: JSON yüklenip dict'e çevrilir, uygulama başında bir kez.
"""
import json
import os
from functools import lru_cache
from langchain_core.tools import tool

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vehicle_catalog.json")


@lru_cache(maxsize=1)
def _load_catalog() -> dict:
    """
    Kataloğu bir kez yükle, in-memory cache'de tut.
    lru_cache(maxsize=1) → uygulama hayatı boyunca sadece bir kez okur.
    """
    with open(CATALOG_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    # Arama kolaylığı için düzleştirilmiş dict: "bmw_3 serisi" → {type, brand, model}
    catalog: dict = {}
    for entry in entries:
        key = f"{entry['brand'].lower()}_{entry['model'].lower()}"
        catalog[key] = entry
        # Variant'ları da ekle
        for variant in entry.get("variants", []):
            vkey = f"{entry['brand'].lower()}_{variant.lower()}"
            catalog[vkey] = {**entry, "model": variant}
    return catalog


@tool
def lookup_vehicle(brand: str, model: str) -> dict:
    """
    Araç marka ve modelinin binek/ticari sınıfını katalogdan döndürür.

    Args:
        brand: Araç markası (örn. "BMW", "Ford")
        model: Araç modeli (örn. "3 Serisi", "Transit")

    Returns:
        {"found": bool, "type": "binek"|"ticari"|None, "brand": str, "model": str}
    """
    catalog = _load_catalog()
    key = f"{brand.lower()}_{model.lower()}"

    if key in catalog:
        entry = catalog[key]
        return {"found": True, "type": entry["type"], "brand": entry["brand"], "model": entry["model"]}

    # Fuzzy: sadece marka eşleşmesi
    brand_matches = [v for k, v in catalog.items() if k.startswith(brand.lower() + "_")]
    if brand_matches:
        return {
            "found": False,
            "type": None,
            "brand": brand,
            "model": model,
            "suggestions": [f"{m['brand']} {m['model']}" for m in brand_matches[:5]],
        }

    return {"found": False, "type": None, "brand": brand, "model": model, "suggestions": []}
