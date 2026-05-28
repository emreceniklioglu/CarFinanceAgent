"""
config.py — Merkezi konfigürasyon dosyası.

Burası tüm çevresel değişkenleri ve model seçimini yönetir.
Model değiştirmek için sadece ACTIVE_MODEL ve ROUTER_MODEL'i güncelle;
başka hiçbir dosyaya dokunman gerekmez.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Model Seçimi ─────────────────────────────────────────────────────────
# Büyük model: FAQ cevaplama, çok adımlı muhakeme
# Küçük model: Intent sınıflandırma, slot çıkarımı, kısa onaylar
ACTIVE_MODEL: str = os.getenv("ACTIVE_MODEL", "gpt-4o")
ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "gpt-4o-mini")

# ── API Key'ler ───────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")

# ── Vektör DB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(__file__), "data", "chroma")
FAQ_COLLECTION_NAME: str = "faq_chunks"

# ── SQLite ────────────────────────────────────────────────────────────────────
DB_PATH: str = os.path.join(os.path.dirname(__file__), "data", "applications.db")

# ── İş Kuralları ─────────────────────────────────────────────────────────────
# Yeni araç
NEW_CAR_MAX_INVOICE: float = 7_000_000.0          # TL — üst fatura limiti
NEW_CAR_FINANCE_RATIO: float = 0.60               # maks finansman / fatura
NEW_CAR_GUARANTOR_THRESHOLD: float = 5_000_000.0  # bu tutarın üstünde kefil zorunlu

# 2. el araç
USED_CAR_MAX_AGE: int = 5                         # yıl
USED_CAR_FINANCE_RATIO: float = 0.40              # maks finansman / kasko
USED_CAR_MAX_FINANCE: float = 3_000_000.0         # TL — mutlak üst limit

# ── Embedding ─────────────────────────────────────────────────────────────────
JINA_EMBEDDING_MODEL: str = "jina-embeddings-v3"
JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"
RAG_TOP_K: int = 3                                # kaç chunk getirilecek

# ── Güvenlik ─────────────────────────────────────────────────────────────────
# Prompt injection tespit için anahtar kelime listesi
INJECTION_KEYWORDS: list[str] = [
    "tüm talimatları unut", "ignore previous", "ignore all", "system:",
    "disregard", "jailbreak", "pretend you are", "roleplay as",
    "önceki talimatları sil", "artık farklı davran",
]

# ── Gradio ───────────────────────────────────────────────────────────────────
GRADIO_TITLE: str = "Araç Finansmanı Chatbot — Agentic Demo"
GRADIO_SHARE: bool = False   # HuggingFace Spaces'te True olur
