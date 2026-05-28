"""
tools/hgs_tool.py — HGS Mock Servisi

AGENTIC PATTERN: Tool Use — Scoped Access
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Çapraz satış agent'ı YALNIZCA bu tool'a erişebilir.
Ön Başvuru DB'ye dokunamaz (least-privilege prensibi).

Gerçek entegrasyonda: Karayolları HGS API çağrısı yapılır.
Burada: in-memory mock (öğrenme amaçlı).
"""
import uuid
import sqlite3
from datetime import datetime
from langchain_core.tools import tool
from config import DB_PATH


@tool
def check_hgs_status(session_id: str) -> dict:
    """
    Müşterinin aktif HGS kaydı var mı kontrol eder.
    Aktif HGS varsa çapraz satış adımı gösterilmez.

    Returns:
        {"has_active_hgs": bool}
    """
    # Mock: session_id'e göre rastgele sonuç (gerçekte CRM'e sorgu)
    # Basitlik için: id'nin son hanesi çifts → HGS var
    has_hgs = (ord(session_id[-1]) % 2 == 0)
    return {"has_active_hgs": has_hgs}


@tool
def register_hgs(session_id: str, application_id: str) -> dict:
    """
    HGS başvurusunu kaydeder.

    Returns:
        {"success": bool, "hgs_id": str, "error": str | None}
    """
    hgs_id = "HGS-" + str(uuid.uuid4())[:6].upper()
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO hgs_registrations VALUES (?,?,?,?,?)",
            (hgs_id, application_id, session_id, datetime.utcnow().isoformat(), "active"),
        )
        conn.commit()
        conn.close()
        return {"success": True, "hgs_id": hgs_id, "error": None}
    except Exception as e:
        return {"success": False, "hgs_id": None, "error": str(e)}
