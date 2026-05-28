"""
setup.py — Kurulum ve İlk Çalıştırma Scripti

Çalıştır:
    python setup.py

Bu script:
1. Gerekli klasörleri oluşturur
2. .env dosyasını kontrol eder
3. SQLite tabloları oluşturur
4. FAQ dokümanını ChromaDB'ye indeksler
5. Temel yapıyı doğrular
"""
import os
import sys

# Proje kök dizini
BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)


def check_env():
    """API key'lerin tanımlı olduğunu kontrol et."""
    from dotenv import load_dotenv
    load_dotenv()

    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        example_path = os.path.join(BASE_DIR, ".env.example")
        print(f"[UYARI] .env dosyası bulunamadı.")
        print(f"        .env.example dosyasını kopyalayın: cp {example_path} {env_path}")
        print(f"        Ardından API key'lerinizi doldurun.\n")
    else:
        print("[✓] .env dosyası bulundu.")

    import config as cfg
    missing = []
    if not cfg.OPENAI_API_KEY and "gpt" in cfg.ACTIVE_MODEL:
        missing.append("OPENAI_API_KEY")
    if not cfg.ANTHROPIC_API_KEY and "claude" in cfg.ACTIVE_MODEL:
        missing.append("ANTHROPIC_API_KEY")
    if not cfg.GOOGLE_API_KEY and "gemini" in cfg.ACTIVE_MODEL:
        missing.append("GOOGLE_API_KEY")
    if not cfg.JINA_API_KEY:
        missing.append("JINA_API_KEY (FAQ RAG için gerekli)")

    if missing:
        print(f"[UYARI] Eksik API key'ler: {', '.join(missing)}")
    else:
        print(f"[✓] API key'ler tanımlı. Aktif model: {cfg.ACTIVE_MODEL}")


def setup_db():
    """SQLite tabloları oluştur."""
    from tools.db_tool import init_db
    init_db()
    print("[✓] SQLite tabloları hazır.")


def index_faq():
    """FAQ dokümanını ChromaDB'ye indeksle."""
    from config import JINA_API_KEY
    if not JINA_API_KEY:
        print("[UYARI] JINA_API_KEY eksik — FAQ indeksleme atlandı.")
        return

    from memory.vector_store import index_faq as _index
    try:
        _index()
        print("[✓] FAQ dokümanı indekslendi.")
    except Exception as e:
        print(f"[HATA] FAQ indeksleme başarısız: {e}")


def verify_catalog():
    """Araç kataloğunun yüklendiğini doğrula."""
    import json
    catalog_path = os.path.join(BASE_DIR, "data", "vehicle_catalog.json")
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"[✓] Araç kataloğu: {len(catalog)} marka/model grubu yüklendi.")


def main():
    print("=" * 50)
    print("Araç Finansmanı Chatbot — Kurulum")
    print("=" * 50 + "\n")

    # Data klasörünü oluştur
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "chroma"), exist_ok=True)

    check_env()
    setup_db()
    verify_catalog()
    index_faq()

    print("\n" + "=" * 50)
    print("Kurulum tamamlandı!")
    print("\nUygulamayı başlatmak için:")
    print("  python app.py")
    print("\nVeya doğrudan Gradio ile:")
    print("  gradio app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
