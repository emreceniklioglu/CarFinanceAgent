"""
graph/workflow.py — LangGraph StateGraph Tanımı

AGENTIC PATTERN: State Machine (LangGraph)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangGraph'ta bir "graph" şunlardan oluşur:
  - StateGraph: state type + node'lar + edge'ler
  - Node: state alan, state parçası döndüren fonksiyon
  - Edge: iki node arasındaki bağlantı
  - Conditional Edge: state'e göre dinamik yönlendirme

compile() çağrısı graph'ı çalıştırılabilir hale getirir.
MemorySaver ile her thread (oturum) kendi checkpoint'ını korur.

Görsel üretmek için:
  from graph.workflow import create_graph
  g = create_graph()
  print(g.get_graph().draw_mermaid())
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import ConversationState
from graph.nodes import (
    input_guard_node,
    supervisor_node,
    intake_new_node,
    intake_used_node,
    faq_agent_node,
    validation_new_node,
    validation_used_node,
    reflection_node,
    recap_node,
    submission_node,
    crosssell_node,
    wait_node,
)
from graph.edges import (
    route_after_supervisor,
    route_after_intake_new,
    route_after_intake_used,
    route_after_validation,
    route_after_reflection,
    route_after_recap,
    route_after_submission,
    route_after_crosssell,
    route_after_faq,
)

# Singleton compiled graph
_compiled_graph = None


def create_graph():
    """
    LangGraph StateGraph'ı oluşturur ve compile eder.

    Graf Yapısı:
      input_guard → supervisor → [intake_new | intake_used | faq_agent]
                                    ↓              ↓
                              validation_new  validation_used
                                    ↓              ↓
                                     reflection
                                         ↓
                                       recap
                                         ↓
                                     submission
                                         ↓
                                      crosssell
                                         ↓
                                        END
    """
    graph = StateGraph(ConversationState)

    # ── Node'ları Ekle ─────────────────────────────────────────────────────────
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("intake_new", intake_new_node)
    graph.add_node("intake_used", intake_used_node)
    graph.add_node("faq_agent", faq_agent_node)
    graph.add_node("validation_new", validation_new_node)
    graph.add_node("validation_used", validation_used_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("recap", recap_node)
    graph.add_node("submission", submission_node)
    graph.add_node("crosssell", crosssell_node)
    graph.add_node("wait_for_input", wait_node)

    # ── Başlangıç Node'u ───────────────────────────────────────────────────────
    # Her kullanıcı mesajı input_guard'dan girer
    graph.set_entry_point("input_guard")

    # ── Sabit Edge'ler (her zaman aynı yönlendirme) ───────────────────────────
    graph.add_edge("input_guard", "supervisor")

    # ── Koşullu Edge'ler (state'e göre dinamik yönlendirme) ──────────────────
    # Supervisor kararı
    # route_after_supervisor(state) fonksiyonu state'e bakar ve bir string döner. 
    # LangGraph bu string'i aşağıdaki dictionary'den lookup'tan geçirip ilgili node'a atlar.
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "intake_new": "intake_new",
            "intake_used": "intake_used",
            "faq_agent": "faq_agent",
            "reflection": "reflection",
            "crosssell": "crosssell",
            "wait_for_input": "wait_for_input",
            "END": END,
        },
    )

    # Yeni araç intake
    graph.add_conditional_edges(
        "intake_new",
        route_after_intake_new,
        {
            "validation_new": "validation_new",
            "wait_for_input": "wait_for_input",
            "END": END,
        },
    )

    # 2. El intake
    graph.add_conditional_edges(
        "intake_used",
        route_after_intake_used,
        {
            "validation_used": "validation_used",
            "wait_for_input": "wait_for_input",
            "END": END,
        },
    )

    # Validation
    graph.add_conditional_edges(
        "validation_new",
        route_after_validation,
        {"reflection": "reflection", "intake_new": "intake_new", "END": END},
    )
    graph.add_conditional_edges(
        "validation_used",
        route_after_validation,
        {"reflection": "reflection", "intake_used": "intake_used", "END": END},
    )

    # Reflection
    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"recap": "recap", "intake_new": "intake_new", "intake_used": "intake_used", "END": END},
    )

    # Recap → Submission
    graph.add_conditional_edges(
        "recap",
        route_after_recap,
        {
            "submission": "submission",
            "intake_new": "intake_new",
            "intake_used": "intake_used",
            "wait_for_input": "wait_for_input",
            "END": END,
        },
    )

    # Submission → CrossSell
    graph.add_conditional_edges(
        "submission",
        route_after_submission,
        {"crosssell": "crosssell", "END": END},
    )

    # CrossSell → END
    graph.add_conditional_edges(
        "crosssell",
        route_after_crosssell,
        {"wait_for_input": "wait_for_input", "END": END},
    )

    # FAQ cevabından kaldığı yere dön
    graph.add_conditional_edges(
        "faq_agent",
        route_after_faq,
        {
            "supervisor": "supervisor",
            "intake_new": "intake_new",
            "intake_used": "intake_used",
            "recap": "recap",
            "wait_for_input": "wait_for_input",
        },
    )

    # Bekleme node'u her zaman durur (kullanıcı girdisi bekliyor)
    graph.add_edge("wait_for_input", END)

    return graph


def get_compiled_graph():
    """
    MemorySaver ile compile edilmiş graph'ı döndürür (singleton).

    MemorySaver: Her thread_id için ayrı checkpoint tutar.
    thread_id = session_id → her kullanıcı oturumu bağımsız.
    """
    global _compiled_graph
    if _compiled_graph is None:
        checkpointer = MemorySaver()
        graph = create_graph()
        _compiled_graph = graph.compile(checkpointer=checkpointer)
    return _compiled_graph


def get_workflow_mermaid() -> str:
    """Graf yapısını Mermaid diyagramı olarak döndürür (görsel için)."""
    graph = create_graph()
    compiled = graph.compile()
    return compiled.get_graph().draw_mermaid()
