"""
tools/db_tool.py — SQLite Başvuru Yazma Aracı

AGENTIC PATTERN: Tool Use — Least-Privilege
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLM doğrudan veritabanına erişemez. Orchestrator bu tool'u çağırır.
TCKN ve hassas veriler maskelenmiş haliyle yazılır.
Sadece submission_agent bu tool'a erişebilir (least-privilege).
"""
import sqlite3
import uuid
from datetime import datetime
from langchain_core.tools import tool
from config import DB_PATH


def init_db():
    """Uygulama başlarken tabloları oluştur (yoksa)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            vehicle_type TEXT,
            created_at TEXT,
            -- Yeni araç alanları
            invoice_amount REAL,
            vehicle_model TEXT,
            requested_amount_new REAL,
            guarantor_tckn_masked TEXT,
            -- 2. el alanları
            kasko_value REAL,
            vehicle_age INTEGER,
            requested_amount_used REAL,
            seller_tckn_masked TEXT,
            -- Meta
            status TEXT DEFAULT 'pending'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            timestamp TEXT,
            event_type TEXT,
            agent TEXT,
            details TEXT,
            masked_input TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hgs_registrations (
            id TEXT PRIMARY KEY,
            application_id TEXT,
            session_id TEXT,
            created_at TEXT,
            status TEXT DEFAULT 'active'
        )
    """)
    conn.commit()
    conn.close()


@tool
def write_application(
    session_id: str,
    vehicle_type: str,
    invoice_amount: float | None = None,
    vehicle_model: str | None = None,
    requested_amount_new: float | None = None,
    guarantor_tckn_masked: str | None = None,
    kasko_value: float | None = None,
    vehicle_age: int | None = None,
    requested_amount_used: float | None = None,
    seller_tckn_masked: str | None = None,
) -> dict:
    """
    Onaylanmış başvuruyu veritabanına yazar.

    Returns:
        {"success": bool, "application_id": str, "error": str | None}
    """
    app_id = str(uuid.uuid4())[:8].upper()  # Kısa, okunabilir başvuru no

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO applications VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                app_id, session_id, vehicle_type,
                datetime.utcnow().isoformat(),
                invoice_amount, vehicle_model, requested_amount_new, guarantor_tckn_masked,
                kasko_value, vehicle_age, requested_amount_used, seller_tckn_masked,
                "pending",
            ),
        )
        conn.commit()
        conn.close()
        return {"success": True, "application_id": app_id, "error": None}
    except Exception as e:
        return {"success": False, "application_id": None, "error": str(e)}


@tool
def get_application_history(session_id: str) -> dict:
    """Bir oturumun geçmiş başvurularını listeler."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, vehicle_type, created_at, status FROM applications WHERE session_id=? ORDER BY created_at DESC",
        (session_id,),
    ).fetchall()
    conn.close()
    return {"applications": [{"id": r[0], "type": r[1], "date": r[2], "status": r[3]} for r in rows]}
