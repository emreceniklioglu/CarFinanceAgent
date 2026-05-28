"""
graph/state.py — Merkezi Konuşma Durumu (State Machine Pattern)

AGENTIC PATTERN: State Machine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangGraph'ta her şey bir "state" etrafında döner. Bu TypedDict:
- Konuşmanın hangi adımında olduğumuzu (current_step)
- Hangi slot'ların dolu olduğunu (invoice_amount, vehicle_model vb.)
- Konuşma geçmişini (messages)
- Hata ve meta bilgilerini
tüm agent'lar arasında paylaşılan tek gerçek kaynaktır.

Her node bu state'i okur, değiştirir ve geri yazar.
LangGraph, iki çağrı arasında state'i otomatik saklar (MemorySaver ile).
"""
from __future__ import annotations
from typing import Literal, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class ConversationState(TypedDict):
    """
    Bir oturumun tüm bilgisini taşıyan merkezi veri yapısı.

    LangGraph bu TypedDict'i her node çağrısında günceller ve
    bir sonraki node'a aktarır. Siz node fonksiyonlarında
    sadece değişen alanları döndürmeniz yeterlidir.
    """

    # ── Oturum Kimliği ────────────────────────────────────────────────────────
    session_id: str                     # her konuşma için benzersiz UUID
    current_step: str                   # hangi graph node'undayız (debug için)

    # ── Araç Türü ─────────────────────────────────────────────────────────────
    vehicle_type: Optional[Literal["new", "used"]]   # None = henüz belirlenmedi

    # ── Yeni Araç Slot'ları (Dal A) ───────────────────────────────────────────
    invoice_amount: Optional[float]     # Proforma fatura tutarı (TL)
    vehicle_model: Optional[str]        # Normalize edilmiş marka + model
    vehicle_model_type: Optional[Literal["binek", "ticari"]]  # katalog sonucu
    requested_amount_new: Optional[float]  # İstenen finansman tutarı
    guarantor_tckn: Optional[str]       # Maskelenmiş kefil TCKN (***7890***)
    kefil_required: bool                # fatura >= 5M ise True

    # ── 2. El Araç Slot'ları (Dal B) ─────────────────────────────────────────
    kasko_value: Optional[float]        # Güncel kasko değeri (TL)
    vehicle_age: Optional[int]          # Araç yaşı (yıl)
    requested_amount_used: Optional[float]  # İstenen finansman tutarı
    seller_tckn: Optional[str]          # Satıcı TCKN (opsiyonel, maskelenmiş)

    # ── Konuşma Geçmişi ───────────────────────────────────────────────────────
    messages: list[BaseMessage]         # tüm kullanıcı + asistan mesajları

    # ── Meta / Akış Kontrolü ──────────────────────────────────────────────────
    application_id: Optional[str]       # Başvuru tamamlandığında UUID
    hgs_consent: Optional[bool]         # HGS çapraz satış kabulü
    errors: list[str]                   # biriken hata mesajları
    flow_ended: bool                    # akış bitti mi (red, tamamlanma)

    # ── FAQ Snapshot ──────────────────────────────────────────────────────────
    # FAQ moduna girerken mevcut adım buraya kaydedilir;
    # FAQ tamamlanınca geri dönülür. Slot'lar SİLİNMEZ.
    faq_snapshot: Optional[dict]        # {"step": "A2", "pending_question": "..."}

    # ── Onay Ekranı ───────────────────────────────────────────────────────────
    recap_confirmed: bool               # müşteri tüm bilgileri onayladı mı
    edit_target_slot: Optional[str]     # düzenlemek istenen alan adı

    # ── Beklenen Slot (intake için kalıcı) ────────────────────────────────────
    # current_step her node'da değişiyor, bu yüzden hangi slot'un cevabını
    # beklediğimizi ayrı bir alanda tutuyoruz. intake_new/used buraya yazar
    # ve bir sonraki turda buradan okur.
    awaited_slot: Optional[str]


def create_initial_state(session_id: str) -> ConversationState:
    """
    Yeni bir oturum için boş başlangıç state'i döndürür.
    Tüm slot'lar None, tüm listeler boş, bool'lar False.
    """
    return ConversationState(
        session_id=session_id,
        current_step="start",
        vehicle_type=None,
        invoice_amount=None,
        vehicle_model=None,
        vehicle_model_type=None,
        requested_amount_new=None,
        guarantor_tckn=None,
        kefil_required=False,
        kasko_value=None,
        vehicle_age=None,
        requested_amount_used=None,
        seller_tckn=None,
        messages=[],
        application_id=None,
        hgs_consent=None,
        errors=[],
        flow_ended=False,
        faq_snapshot=None,
        recap_confirmed=False,
        edit_target_slot=None,
        awaited_slot=None,
    )
