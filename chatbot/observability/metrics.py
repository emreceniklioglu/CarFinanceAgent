"""
observability/metrics.py — İş ve Teknik Metrikler

REPORT Karşılığı: "Metric / Alert" (Gözlem & Denetim)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audit log tablosundan gerçek zamanlı metrik hesaplamaları yapar.
Dashboard bu modülü sorguladığında güncel sayılar döner.

Teknik metrikler:
  - LLM latency (p50, p95 hesaplaması)
  - Token kullanımı (model bazında)

İş metrikleri:
  - Başvuru tamamlama oranı
  - Adım bazlı terk (en çok hangi adımda bırakılıyor)
  - Guard tetiklenme sayısı (injection girişimi vb.)
  - HGS dönüşüm oranı
  - FAQ kullanım sıklığı
"""
import json
import sqlite3
from datetime import datetime, timezone
from config import DB_PATH


def _fmt_local_time(ts: str) -> str:
    """ISO UTC timestamp'i yerel saate çevirip HH:MM:SS döndürür."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%H:%M:%S")
    except Exception:
        return ts[11:19] if len(ts) >= 19 else ts


def get_summary_metrics() -> dict:
    """Dashboard için özet metrik paketi."""
    conn = sqlite3.connect(DB_PATH)

    # Toplam başvuru sayısı
    total_apps = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]

    # HGS kayıt sayısı
    hgs_count = conn.execute("SELECT COUNT(*) FROM hgs_registrations").fetchone()[0]

    # Guard tetiklenme sayısı
    guard_triggers = conn.execute(
        "SELECT COUNT(*) FROM audit_events WHERE event_type='guard_event' AND details LIKE '%\"triggered\": true%'"
    ).fetchone()[0]

    # Son 100 LLM çağrısından latency hesapla
    latencies = conn.execute(
        "SELECT details FROM audit_events WHERE event_type='llm_call' ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()

    lat_values = []
    total_tokens_in = 0
    total_tokens_out = 0
    for (det,) in latencies:
        d = json.loads(det)
        if "latency_ms" in d:
            lat_values.append(d["latency_ms"])
        total_tokens_in += d.get("tokens_in", 0)
        total_tokens_out += d.get("tokens_out", 0)

    lat_values.sort()
    p50 = lat_values[len(lat_values) // 2] if lat_values else 0
    p95 = lat_values[int(len(lat_values) * 0.95)] if lat_values else 0

    # Araç türü dağılımı
    type_dist = conn.execute(
        "SELECT vehicle_type, COUNT(*) FROM applications GROUP BY vehicle_type"
    ).fetchall()

    conn.close()

    return {
        "total_applications": total_apps,
        "hgs_registrations": hgs_count,
        "hgs_conversion_rate": round(hgs_count / total_apps * 100, 1) if total_apps > 0 else 0,
        "guard_triggers": guard_triggers,
        "llm_latency_p50_ms": p50,
        "llm_latency_p95_ms": p95,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "vehicle_type_distribution": {r[0]: r[1] for r in type_dist},
    }


def get_audit_table(limit: int = 30) -> list[dict]:
    """Dashboard audit log tablosu için son olaylar."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT session_id, timestamp, event_type, agent, details
           FROM audit_events ORDER BY timestamp DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()

    result = []
    for r in rows:
        details = json.loads(r[4])
        # Latency veya triggered gibi özet bilgi çıkar
        summary = ""
        if "latency_ms" in details:
            summary = f"{details['latency_ms']}ms"
        elif "triggered" in details:
            summary = "✓ Tetiklendi" if details["triggered"] else "- Geçti"
        elif "slot" in details:
            summary = f"{details['slot']}: {details.get('new')}"
        elif "from" in details and "to" in details:
            summary = f"{details['from']} → {details['to']}"
            if details.get("reason"):
                summary += f" ({details['reason']})"

        prompt_preview = _preview(details.get("prompt"))
        response_preview = _preview(details.get("response"))

        result.append({
            "session": r[0][:8] + "...",
            "time": _fmt_local_time(r[1]),
            "type": r[2],
            "agent": r[3],
            "summary": summary,
            "prompt": prompt_preview,
            "response": response_preview,
        })
    return result


def _preview(text, limit: int = 200) -> str:
    if text is None:
        return ""
    s = str(text).replace("\n", " ⏎ ")
    return s if len(s) <= limit else s[:limit] + "…"


def get_llm_calls(limit: int = 50, session_id: str | None = None) -> list[dict]:
    """LLM çağrılarını prompt+response ile birlikte döndür (izleme için)."""
    conn = sqlite3.connect(DB_PATH)
    if session_id:
        rows = conn.execute(
            """SELECT session_id, timestamp, agent, details FROM audit_events
               WHERE event_type='llm_call' AND session_id=? ORDER BY timestamp DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT session_id, timestamp, agent, details FROM audit_events
               WHERE event_type='llm_call' ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    conn.close()

    out = []
    for sid, ts, agent, det in rows:
        d = json.loads(det)
        out.append({
            "session": sid[:8] + "...",
            "time": _fmt_local_time(ts),
            "agent": agent,
            "model": d.get("model", ""),
            "latency_ms": d.get("latency_ms", 0),
            "tokens_in": d.get("tokens_in", 0),
            "tokens_out": d.get("tokens_out", 0),
            "prompt": d.get("prompt", ""),
            "response": d.get("response", ""),
        })
    return out
