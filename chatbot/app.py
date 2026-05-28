"""
app.py — Ana Uygulama Girişi (Gradio + FastAPI)

Gradio ChatInterface ile sohbet arayüzü sunar.
Üst kısımda model seçici dropdown ile GPT/Claude/Gemini arası geçiş.
Alt sekmede audit/metrik dashboard.

HuggingFace Spaces için: bu dosyayı root'ta tut, demo.launch() çağır.
"""
import uuid
import os
import sys
import io

# Proje kök dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(__file__))

# Windows konsolunda emoji/unicode karakterlerin print hatası vermemesi için
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import gradio as gr
from langchain_core.messages import HumanMessage

from config import GRADIO_TITLE, GRADIO_SHARE
from graph.state import create_initial_state
from graph.workflow import get_compiled_graph, get_workflow_mermaid
from memory.checkpointer import get_thread_config
from memory.vector_store import index_faq
from tools.db_tool import init_db
from observability.dashboard import build_dashboard_tab

# ── Başlangıç: DB + FAQ İndeksleme ───────────────────────────────────────────
init_db()
try:
    index_faq()
except Exception as e:
    print(f"[Uyarı] FAQ indeksleme atlandı: {e}")

# Aktif oturumlar: session_id → ConversationState
_sessions: dict = {}


def chat(message: str, history: list, session_id: str, model_choice: str) -> str:
    """
    Gradio ChatInterface'in çağırdığı ana fonksiyon.

    Args:
        message: Kullanıcının yazdığı metin
        history: Önceki mesajlar (Gradio formatı)
        session_id: Oturum UUID (Gradio state'te saklanır)
        model_choice: Kullanıcının seçtiği model

    Returns:
        Chatbot'un cevabı (string)
    """
    # Model seçimini config'e yansıt (runtime değişiklik)
    _apply_model_choice(model_choice)

    compiled = get_compiled_graph()
    config = get_thread_config(session_id)

    # Oturum state'i yoksa oluştur
    if session_id not in _sessions:
        _sessions[session_id] = create_initial_state(session_id)

    state = _sessions[session_id]

    # Kullanıcı mesajını state'e ekle
    updated_messages = state["messages"] + [HumanMessage(content=message)]
    state = {**state, "messages": updated_messages}
    print(f"[DEBUG current state before invoke] {state}", flush=True)
    import sys
    print(f"[DEBUG chat] sid={session_id[:8]} msg={message!r} awaited={state.get('awaited_slot')} vtype={state.get('vehicle_type')} step={state.get('current_step')}", flush=True)
    sys.stdout.flush()

    # Graph'ı çalıştır
    try:
        result = compiled.invoke(state, config=config)
        _sessions[session_id] = result
        print(f"[DEBUG chat AFTER] awaited={result.get('awaited_slot')} vtype={result.get('vehicle_type')} step={result.get('current_step')} invoice={result.get('invoice_amount')}", flush=True)
        sys.stdout.flush()

        # Son AI mesajını bul
        ai_messages = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
        if ai_messages:
            return ai_messages[-1].content
        return "İşleniyor..."
    except Exception as e:
        print(f"[ERROR chat] sid={session_id[:8]} {type(e).__name__}: {e}", flush=True)
        from litellm.exceptions import ServiceUnavailableError, RateLimitError, Timeout
        if isinstance(e, (ServiceUnavailableError, RateLimitError, Timeout)):
            return ("Sistemimiz şu anda yoğun. Lütfen birkaç saniye sonra "
                    "mesajınızı tekrar gönderir misiniz? 🙏")
        return "Beklenmeyen bir sorun oluştu. Lütfen tekrar deneyin."


def new_session() -> tuple[str, list]:
    """Yeni oturum başlat — session_id üret, history temizle."""
    session_id = str(uuid.uuid4())
    welcome = "Merhaba! 👋 Araç finansmanı ön başvurusu için yardımcı olabilirim.\nYeni araç mı, 2. el araç mı düşünüyorsunuz?"
    return session_id, [(None, welcome)]


def show_workflow() -> str:
    """Workflow Mermaid diyagramını döndürür."""
    try:
        return f"```mermaid\n{get_workflow_mermaid()}\n```"
    except Exception as e:
        return f"Diyagram üretilirken hata: {e}"


def _apply_model_choice(choice: str):
    """
    Model seçimini runtime'da config modülüne uygular.
    llm/factory.py config.ACTIVE_MODEL'i her çağrıda taze okuduğu için
    bu değişiklik anında tüm agent'lara yansır — oturum sıfırlanmaz.
    """
    import config as cfg
    model_map = {
        "GPT-4o":           ("gpt-4o",                        "gpt-4o-mini"),
        "GPT-4o Mini":      ("gpt-4o-mini",                   "gpt-4o-mini"),
        "Claude Sonnet":    ("claude-sonnet-4-6",              "claude-haiku-4-5-20251001"),
        "Claude Haiku":     ("claude-haiku-4-5-20251001",      "claude-haiku-4-5-20251001"),
        "Gemini Flash":     ("gemini/gemini-2.5-flash",      "gemini/gemini-2.5-flash-lite"),
        "Gemini Flash Lite":("gemini/gemini-2.5-flash-lite", "gemini/gemini-2.5-flash-lite"),
    }
    if choice in model_map:
        cfg.ACTIVE_MODEL, cfg.ROUTER_MODEL = model_map[choice]


# ── Gradio UI ─────────────────────────────────────────────────────────────────

MODEL_CHOICES = ["GPT-4o", "GPT-4o Mini", "Claude Sonnet", "Claude Haiku", "Gemini Flash", "Gemini Flash Lite"]

# Config'deki aktif modele göre dropdown başlangıç değerini belirle
_REVERSE_MODEL_MAP = {
    "gpt-4o":                          "GPT-4o",
    "gpt-4o-mini":                     "GPT-4o Mini",
    "claude-sonnet-4-6":               "Claude Sonnet",
    "claude-haiku-4-5-20251001":       "Claude Haiku",
    "gemini/gemini-2.5-flash":         "Gemini Flash",
    "gemini/gemini-2.5-flash-lite":    "Gemini Flash Lite",
    "gemini/gemini-2.0-flash":         "Gemini Flash",
    "gemini/gemini-flash-latest":      "Gemini Flash",
    "gemini/gemini-2.0-flash-latest":  "Gemini Flash",
    "gemini/gemini-2.0-flash-lite":    "Gemini Flash Lite",
}

import config as _init_cfg
_DEFAULT_MODEL_CHOICE = _REVERSE_MODEL_MAP.get(_init_cfg.ACTIVE_MODEL, "GPT-4o")

with gr.Blocks(title=GRADIO_TITLE) as demo:
    gr.Markdown(f"# {GRADIO_TITLE}")
    gr.Markdown("> Agentic AI öğrenme demosu — LangGraph + LiteLLM + ChromaDB/Jina RAG")

    # Oturum state'i (tarayıcıda saklanır)
    session_state = gr.State(value=str(uuid.uuid4()))

    with gr.Tab("💬 Chatbot"):
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=_DEFAULT_MODEL_CHOICE,
                label="🤖 Dil Modeli",
                scale=2,
            )
            import config as _cfg
            active_model_label = gr.Textbox(
                value=f"{_cfg.ACTIVE_MODEL}  |  router: {_cfg.ROUTER_MODEL}",
                label="Aktif Model",
                interactive=False,
                scale=3,
            )
            new_session_btn = gr.Button("🔄 Yeni Oturum", scale=1)

        chatbot_ui = gr.Chatbot(
            label="Arac Finansmani Chatbot",
            height=500,
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Mesajınızı yazın... (örn: yeni araç finansmanı istiyorum)",
                label="",
                scale=5,
                container=False,
            )
            send_btn = gr.Button("Gönder", scale=1, variant="primary")

        # Model değişince etiketi güncelle (oturum SIFIRLANMAZ)
        def on_model_change(choice):
            _apply_model_choice(choice)
            import config as cfg
            return f"{cfg.ACTIVE_MODEL}  |  router: {cfg.ROUTER_MODEL}"

        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[active_model_label],
        )

        # Mesaj gönderme
        def respond(message, history, session_id, model):
            if not message.strip():
                return history, ""
            response = chat(message, history, session_id, model)
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
            return history, ""

        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot_ui, session_state, model_dropdown],
            outputs=[chatbot_ui, msg_input],
        )
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot_ui, session_state, model_dropdown],
            outputs=[chatbot_ui, msg_input],
        )

        # Yeni oturum
        def start_new(old_sid):
            new_sid = str(uuid.uuid4())
            welcome = "Merhaba! 👋 Araç finansmanı ön başvurusu için yardımcı olabilirim.\nYeni araç mı, 2. el araç mı düşünüyorsunuz?"
            return new_sid, [{"role": "assistant", "content": welcome}]

        new_session_btn.click(
            fn=start_new,
            inputs=[session_state],
            outputs=[session_state, chatbot_ui],
        )

    with gr.Tab("🗺️ Workflow Diyagramı"):
        gr.Markdown("LangGraph state machine görselleştirmesi:")
        diagram_output = gr.Markdown()
        show_btn = gr.Button("Diyagramı Göster")
        show_btn.click(fn=show_workflow, outputs=diagram_output)

    # Audit/Metrik Dashboard sekmesi
    build_dashboard_tab()

    with gr.Tab("ℹ️ Agentic Pattern'ler"):
        gr.Markdown("""
## Bu Projede Kullanılan Agentic Pattern'ler

| Pattern | Dosya | Açıklama |
|---|---|---|
| **State Machine** | `graph/workflow.py` | LangGraph StateGraph — tüm akış |
| **Supervisor** | `agents/supervisor.py` | Intent sınıflandırma + dal yönlendirme |
| **Tool Use** | `tools/` | @tool ile iş kuralları kapsülleme |
| **ReAct** | `agents/validation_agent.py` | Reason + Act döngüsü |
| **RAG** | `agents/faq_agent.py` | Jina embed + ChromaDB retrieval |
| **Reflection** | `agents/reflection_agent.py` | Self-check onay öncesi |
| **Short-Term Memory** | `memory/checkpointer.py` | LangGraph MemorySaver |
| **Long-Term Memory** | `memory/vector_store.py` | ChromaDB persist |
| **Least-Privilege** | `agents/crosssell_agent.py` | Scoped tool access |
| **Input/Output Guard** | `guardrails/` | PII + injection filter |
| **Audit Log** | `observability/audit_logger.py` | KVKK/BDDK uyumlu log |
| **Streaming** | `app.py` | Gradio token streaming |
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=GRADIO_SHARE,
    )
