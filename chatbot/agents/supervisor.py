"""
agents/supervisor.py — Supervisor / Router Agent

AGENTIC PATTERN: Supervisor Pattern
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Supervisor pattern'de bir üst agent, gelen isteği analiz eder
ve hangi alt-agent'ın görevi üstleneceğine karar verir.

Bu agent:
1. Müşterinin mesajını alır
2. Küçük (router) modeli kullanarak intent sınıflandırır
3. Confidence < 0.7 ise netleştirme sorusu üretir
4. Sonucu ConversationState'e yazar → LangGraph graph edge'i
   bu sonuca göre bir sonraki node'u seçer

NEDEN Supervisor Pattern?
- Her alt-agent kendi sorumluluğuna odaklanır (single responsibility)
- Yeni bir dal (örn. "sigorta bilgisi") eklemek için sadece Supervisor
  ve yeni agent yazılır; diğer agent'lara dokunulmaz
- Supervisor'ın kararı loglarda görülür → debug kolaylığı
"""
import time
from langchain_core.messages import HumanMessage, AIMessage
from llm.factory import get_router_llm
from llm.json_utils import parse_llm_json
from llm.prompts import SUPERVISOR_SYSTEM, MSG_OUT_OF_SCOPE
from graph.state import ConversationState
from observability.audit_logger import log_llm_call


def run_supervisor(state: ConversationState) -> dict:
    """
    Supervisor node fonksiyonu.
    LangGraph bu fonksiyonu çağırır; dönen dict state'i günceller.

    Returns: state güncellemesi (sadece değişen alanlar)
    """
    last_message = state["messages"][-1]
    user_text = last_message.content if hasattr(last_message, "content") else str(last_message)

    # ── Mid-flow bypass ──────────────────────────────────────────────────────
    # Zaten bir intake/recap/crosssell akışındaysak LLM sınıflandırması yapmadan
    # mevcut adımı koru. Aksi halde "onaylıyorum", "2.500.000" gibi cevaplar
    # yanlış intent (unclear / out_of_scope) olarak yorumlanır ve akış kopar.
    # NOT: input_guard node'u current_step'i "input_guard_passed" olarak ezdiği
    # için step kontrolüne güvenmiyoruz; vehicle_type + doldurulmuş slot'lara
    # veya awaited_slot'a bakıyoruz.
    vtype = state.get("vehicle_type")
    has_new_progress = vtype == "new" and (
        state.get("invoice_amount")
        or state.get("vehicle_model")
        or state.get("requested_amount_new")
    )
    has_used_progress = vtype == "used" and (
        state.get("kasko_value")
        or state.get("vehicle_age")
        or state.get("requested_amount_used")
    )
    in_mid_flow = state.get("awaited_slot") or has_new_progress or has_used_progress

    if in_mid_flow and vtype in ("new", "used"):
        # Soru soruyorsa mid-flow'da bile FAQ'e yönlendir
        if _looks_like_finance_question(user_text):
            return {
                "current_step": "supervisor_routed_faq",
                "vehicle_type": vtype,
                "_supervisor_intent": "faq",
            }
        intent = "new_car" if vtype == "new" else "used_car"
        return {
            "current_step": f"supervisor_routed_{intent}",
            "vehicle_type": vtype,
            "_supervisor_intent": intent,
        }

    # Deterministik kestirme: akış başında net araç türü ifadeleri LLM'e gitmeden yakalanır.
    # (LLM router "yeni araç" gibi kısa ifadeleri bazen belirsiz sayıp clarifying'e düşürüyor.)
    lower_text = user_text.lower().strip()
    if not state.get("vehicle_type") and not _looks_like_finance_question(user_text):
        is_new = any(w in lower_text for w in ["yeni", "sıfır", "sifir", "0 km", "0km", "sfr"])
        is_used = any(w in lower_text for w in ["2.el", "2. el", "2el", "ikinci el", "kullanılmış", "kullanilmis"])
        if is_new and not is_used:
            return {"current_step": "supervisor_routed_new_car", "vehicle_type": "new", "_supervisor_intent": "new_car"}
        if is_used and not is_new:
            return {"current_step": "supervisor_routed_used_car", "vehicle_type": "used", "_supervisor_intent": "used_car"}

    llm = get_router_llm()
    start = time.time()

    response = llm.invoke([
        {"role": "system", "content": SUPERVISOR_SYSTEM},
        {"role": "user", "content": user_text},
    ])

    latency = int((time.time() - start) * 1000)
    log_llm_call(
        session_id=state["session_id"],
        agent="supervisor",
        model=llm.model,
        latency_ms=latency,
        tokens_in=len(user_text.split()),  # yaklaşık
        tokens_out=len(response.content.split()),
        prompt=[
            {"role": "system", "content": SUPERVISOR_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        response=response.content,
    )

    # JSON parse (markdown code fence ve ek metni tolere eder)
    parsed = parse_llm_json(response.content)
    if parsed:
        intent = parsed.get("intent", "out_of_scope")
        confidence = parsed.get("confidence", 1.0)
        clarification = parsed.get("clarification")
    else:
        intent = "out_of_scope"
        confidence = 0.0
        clarification = None

    # Defansif heuristic — küçük router modeli finansman sorularını bazen yanlış
    # ("out_of_scope", "unclear") ya da düşük confidence ile işaretliyor.
    # Mesaj net bir araç finansmanı sorusuysa faq'e yönlendir ve clarifying'e
    # düşmesini engelle (confidence'ı eşiğin üstüne çek).
    if _looks_like_finance_question(user_text) and intent in ("out_of_scope", "unclear", "faq"):
        intent = "faq"
        confidence = max(confidence, 0.75)

    # Netleştirme gerekiyorsa → mesaj ekle, intent belirsiz bırak
    if intent == "unclear" or confidence < 0.7:
        clarification_text = clarification or "İsteğinizi tam olarak anlamadım. Araç finansmanı mı yoksa başka bir konuda mı yardımcı olayım?"
        return {
            "current_step": "clarifying",
            "messages": state["messages"] + [AIMessage(content=clarification_text)],
        }

    # Intent → vehicle_type mapping
    vehicle_type = None
    if intent == "new_car":
        vehicle_type = "new"
    elif intent == "used_car":
        vehicle_type = "used"

    # Araç türü belirsizse (new_car/used_car ama hangi tür belli değil) → sor
    if intent in ("new_car", "used_car") and vehicle_type and not state.get("vehicle_type"):
        # Mesajda açıkça "yeni"/"sıfır" veya "2.el"/"ikinci el" geçmiyorsa sor
        lower_text = user_text.lower()
        has_new = any(w in lower_text for w in ["yeni", "sıfır", "sfr", "0km", "sifir"])
        has_used = any(w in lower_text for w in ["2.el", "ikinci el", "2el", "kullanılmış", "kullanilmis", "2. el"])
        if not has_new and not has_used:
            return {
                "current_step": "clarifying",
                "messages": state["messages"] + [AIMessage(content="Yeni araç mı, 2. el araç mı düşünüyorsunuz?")],
            }

    # Zaten devam eden bir intake adımındaysak current_step'e dokunma
    # (intake_new/used hangi slot'u beklediğini current_step ile takip eder)
    existing_step = state.get("current_step", "")
    mid_intake = existing_step.startswith("asking_") or existing_step.endswith("_collected")
    new_step = existing_step if mid_intake else f"supervisor_routed_{intent}"

    return {
        "current_step": new_step,
        "vehicle_type": vehicle_type if vehicle_type else state.get("vehicle_type"),
        # Kapsam dışı mesaj varsa hemen yanıtla
        "messages": (
            state["messages"] + [AIMessage(content=MSG_OUT_OF_SCOPE)]
            if intent == "out_of_scope"
            else state["messages"]
        ),
        "flow_ended": intent == "out_of_scope",
        "_supervisor_intent": intent,
    }


_FINANCE_KEYWORDS = (
    "finansman", "kredi", "fatura", "kasko", "kefil", "vade", "faiz", "taksit",
    "hgs", "araç", "arac", "model", "yaş", "yas", "tckn", "sigorta", "ödeme",
    "odeme", "evrak", "belge", "başvuru", "basvuru", "limit", "proforma", "satıcı", "satici",
)
# "milyon", "tl", "tutar", "bin tl" kaldırıldı — saf sayı ifadelerinde false positive yaratıyor
_QUESTION_MARKERS = (
    "?", " mi ", " mı ", " mu ", " mü ", "miyim", "mıyım", "muyum", "müyüm",
    " mi?", " mı?", " mu?", " mü?",
    "nedir", "ne kadar", "kaç", "nasıl", "ne olur", "var mı", "var mi",
)


def _looks_like_finance_question(text: str) -> bool:
    """Mesaj araç finansmanı bağlamında bir soru gibi mi görünüyor?"""
    if not text:
        return False
    lower = text.lower() + " "  # sondaki boşluk: "...mu" gibi cümle sonu eklerini yakala
    has_question = any(m in lower for m in _QUESTION_MARKERS)
    has_finance = any(k in lower for k in _FINANCE_KEYWORDS)
    return has_question and has_finance
