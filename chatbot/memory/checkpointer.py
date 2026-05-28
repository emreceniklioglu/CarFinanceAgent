"""
memory/checkpointer.py — LangGraph MemorySaver Konfigürasyonu

AGENTIC PATTERN: Short-Term Memory (Checkpointing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangGraph MemorySaver, her oturum (thread_id) için state'i
in-memory olarak saklar. Kullanıcı yeni mesaj gönderdiğinde
graph, önceki state'ten devam eder — slot'lar kaybolmaz.

thread_id = session_id: Her kullanıcı oturumu tamamen bağımsız.
İki oturumun state'i birbirine karışamaz.

Üretimde: SqliteSaver veya RedisSaver kullanılır.
Öğrenme için: MemorySaver yeterli.
"""
from langgraph.checkpoint.memory import MemorySaver

# Uygulama genelinde tek bir checkpointer instance
_checkpointer = MemorySaver()


def get_checkpointer() -> MemorySaver:
    """Global checkpointer instance'ını döndürür."""
    return _checkpointer


def get_thread_config(session_id: str) -> dict:
    """
    LangGraph invoke/stream için gerekli config dict'ini döndürür.

    Kullanım:
        graph.invoke(state, config=get_thread_config(session_id))
    """
    return {"configurable": {"thread_id": session_id}}
