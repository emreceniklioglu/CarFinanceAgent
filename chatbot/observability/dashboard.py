"""
observability/dashboard.py — Gradio Audit/Metrik Dashboard Sekmesi

REPORT Karşılığı: Metric/Alert dashboard görselleştirmesi.
Gradio'nun Tab bileşeni ile ana chat'in yanında ikinci bir sekme olarak gösterilir.
"""
import gradio as gr
from observability.metrics import get_summary_metrics, get_audit_table, get_llm_calls


def build_dashboard_tab():
    """
    Gradio Tab bileşeni döndürür — app.py'deki Blocks içine eklenir.
    """
    with gr.Tab("📊 İzleme Paneli"):
        gr.Markdown("## Gerçek Zamanlı Metrikler")

        with gr.Row():
            total_apps = gr.Number(label="Toplam Başvuru", interactive=False)
            hgs_rate = gr.Number(label="HGS Dönüşüm %", interactive=False)
            guard_count = gr.Number(label="Guard Tetiklenme", interactive=False)
            p50 = gr.Number(label="LLM Latency p50 (ms)", interactive=False)
            p95 = gr.Number(label="LLM Latency p95 (ms)", interactive=False)

        refresh_btn = gr.Button("🔄 Yenile")
        audit_table = gr.Dataframe(
            headers=["Oturum", "Saat", "Tür", "Agent", "Özet", "Prompt", "Response"],
            label="Son Denetim Olayları",
            interactive=False,
            wrap=True,
        )

        gr.Markdown("## 🤖 LLM Çağrıları (prompt + response)")
        llm_session_filter = gr.Textbox(
            label="Oturum filtresi (boş = tümü, kısmî UUID kabul)",
            placeholder="örn: 3f9a1b2c",
        )
        llm_table = gr.Dataframe(
            headers=["Oturum", "Saat", "Agent", "Model", "Latency (ms)", "Tok In", "Tok Out", "Prompt", "Response"],
            label="Son LLM Çağrıları",
            interactive=False,
            wrap=True,
        )

        def refresh_metrics(sid_filter: str):
            m = get_summary_metrics()
            rows = get_audit_table(30)
            table_data = [
                [r["session"], r["time"], r["type"], r["agent"], r["summary"],
                 r.get("prompt", ""), r.get("response", "")]
                for r in rows
            ]

            sid = sid_filter.strip() or None
            llm_rows = get_llm_calls(50, session_id=None)
            if sid:
                llm_rows = [r for r in llm_rows if sid in r["session"]]
            llm_data = [
                [r["session"], r["time"], r["agent"], r["model"],
                 r["latency_ms"], r["tokens_in"], r["tokens_out"],
                 r["prompt"], r["response"]]
                for r in llm_rows
            ]

            return (
                m["total_applications"],
                m["hgs_conversion_rate"],
                m["guard_triggers"],
                m["llm_latency_p50_ms"],
                m["llm_latency_p95_ms"],
                table_data,
                llm_data,
            )

        refresh_btn.click(
            fn=refresh_metrics,
            inputs=[llm_session_filter],
            outputs=[total_apps, hgs_rate, guard_count, p50, p95, audit_table, llm_table],
        )

    return refresh_btn
