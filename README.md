---
title: CarFinanceAgent
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.10.0
app_file: chatbot/app.py
python_version: "3.11"
pinned: false
---
Huggingface space : https://huggingface.co/spaces/emrecn/CarFinanceAgent
# Araç Finansmanı Chatbot — Agentic Demo

Bankacılık mobil uygulaması için geliştirilmiş **LangGraph tabanlı çok-ajanlı (multi-agent)** araç finansmanı ön başvuru chatbotu. Müşteri, konuşma arayüzü üzerinden yeni veya ikinci el araç finansmanı ön başvurusunu tamamen otomatik olarak tamamlayabilir; süreç boyunca SSS sorabilir, verilerini düzeltebilir ve çapraz satış teklifini kabul edebilir.

---

## İçindekiler

1. [Proje Kapsamı](#1-proje-kapsamı)
2. [İş Kuralları](#2-iş-kuralları)
3. [Agentic Workflow](#3-agentic-workflow)
4. [Agent'lar](#4-agentlar)
5. [Tool'lar (Araçlar)](#5-toollar-araçlar)
6. [Güvenlik Katmanları](#6-güvenlik-katmanları)
7. [Teknolojiler](#7-teknolojiler)
8. [Proje Yapısı](#8-proje-yapısı)
9. [İzleme Paneli](#9-i̇zleme-paneli)
10. [Kurulum ve Çalıştırma](#10-kurulum-ve-çalıştırma)
11. [Konfigürasyon](#11-konfigürasyon)
12. [Diyagramlar](#12-diyagramlar)

---

## 1. Proje Kapsamı

| Kapsam İçi | Kapsam Dışı |
|---|---|
| Yeni araç finansman ön başvurusu | Nihai kredi onayı |
| 2. el araç finansman ön başvurusu | Faiz oranı garantisi |
| SSS / genel bilgi sorguları (RAG) | Ticari araç finansmanı |
| HGS çapraz satış teklifi | Bireysel kredi, konut kredisi vb. |
| Hedefli alan güncellemesi | Şube işlemleri |
| Başvuru numarası oluşturma | Gerçek kredi skorlama |

Sistem, müşteriye **adım adım rehberlik** eder. LLM her cevabı üretmez; deterministik iş kuralları önceliklidir ve LLM yalnızca niyet sınıflandırma, slot çıkarımı ile FAQ cevaplama için kullanılır.

---

## 2. İş Kuralları

### 2.1 Yeni Araç Finansmanı

| Kural | Değer |
|---|---|
| Proforma fatura üst limiti | **7.000.000 TL** |
| Maksimum finansman oranı | Fatura tutarının **%60'ı** |
| Kefil zorunluluğu eşiği | Fatura **≥ 5.000.000 TL** ise kefil TCKN zorunlu |
| Araç tipi kısıtı | Yalnızca **binek** araç — ticari araçlar reddedilir |
| Limit aşımı davranışı | Sistem maksimum tutarı önerir, müşteriden yeni değer ister |

### 2.2 İkinci El Araç Finansmanı

| Kural | Değer |
|---|---|
| Maksimum araç yaşı | **5 yıl** |
| Maksimum finansman oranı | Kasko değerinin **%40'ı** |
| Mutlak finansman üst limiti | **3.000.000 TL** |
| Uygulanan limit | `min(%40 × kasko, 3.000.000 TL)` |
| Satıcı TCKN | Opsiyonel (müşteri vermek istemeyebilir) |

### 2.3 Genel Kurallar

- **TCKN format kontrolü** intake aşamasında regex ile yapılır: 11 hane, yalnızca rakam, sıfırla başlamaz. Mod-11 doğrulama fonksiyonu (`validate_tckn`) araç olarak kodda hazır fakat validation agent tarafından şu an çağrılmıyor.
- Tüm TCKN'ler veritabanına yazılmadan önce **maskelenir** (`123****8901` formatı).
- Müşteri onay ekranında her alanı **hedefli olarak değiştirebilir**; yalnızca değiştirilen alan ve ona bağımlı alanlar yeniden sorulur.
- Başvuru tamamlandıktan sonra UUID formatında **başvuru numarası** üretilir.
- HGS çapraz satış yalnızca **aktif HGS'i olmayan** müşterilere sunulur.

---

## 3. Agentic Workflow

Sistem **Supervisor + Specialist Agents** mimarisi üzerine inşa edilmiştir.

```
Müşteri Mesajı
       │
       ▼
  [Input Guard]  ← PII maskeleme, injection tespiti
       │
       ▼
  [Supervisor]   ← Niyet sınıflandırma, yönlendirme
       │
  ┌────┴─────────────────────────────────────────┐
  │           │             │           │         │
  ▼           ▼             ▼           ▼         ▼
[Intake    [Intake      [FAQ/RAG   [Submission [Cross-sell
 Yeni]      2.El]        Agent]     Agent]     Agent HGS]
  │           │
  └────┬──────┘
       ▼
[Validation Agent]  ← TCKN, katalog, iş kuralı kontrolleri
       │
       ▼
  [Output Guard]  ← PII redaction, finansal taahhüt filtresi
       │
       ▼
  Müşteri Cevabı
```

### Paralel FAQ Akışı

Müşteri herhangi bir adımda SSS sorarsa:
1. Supervisor `intent=faq` tespit eder
2. Mevcut adım bilgisi state içindeki `faq_snapshot` alanına kaydedilir (`{"step": current_step}`)
3. FAQ Agent soruyu yanıtlar
4. Oturum önceki adımdan **devam eder** — müşteri yerini kaybetmez

### Hedefli Güncelleme Akışı

Onay ekranında müşteri alan değiştirmek istediğinde:
1. LLM hangi alanın değiştirileceğini çıkarır
2. Yalnızca o alan yeniden sorulur
3. Bağımlı alanlar bozulduysa otomatik uyarı verilir

---

## 4. Agent'lar

| Agent | Sorumluluk | Agentic Desen |
|---|---|---|
| **Supervisor Agent** | Niyet sınıflandırma, yönlendirme, planlama | Planner + Router |
| **Intake Agent (Yeni Araç)** | Fatura, model, finansman, kefil TCKN toplanması; araç kataloğu sorgusu (binek/ticari doğrulama) | Slot Filling + Tool Use |
| **Intake Agent (2. El)** | Kasko, yaş, finansman, satıcı TCKN toplanması | Slot Filling + Tool Use |
| **Validation Agent** | Finansman limit kontrolü, iş kuralı kontrolleri | Tool Use (deterministik) |
| **FAQ/RAG Agent** | SSS sorularını vektör DB'den yanıtlama | RAG + Tool Use |
| **Submission Agent** | Onaylanan başvuruyu SQLite'a kaydetme | Tool Use + Write |
| **Cross-sell Agent** | HGS durumu sorgulama ve kayıt | Tool Use (Scoped Access) |

Her agent **least-privilege** prensibine göre çalışır: yalnızca ihtiyaç duyduğu tool'lara erişimi vardır.

---

## 5. Tool'lar (Araçlar)

| Tool | Dosya | Açıklama |
|---|---|---|
| `validate_tckn` / `_mask` | `tools/tckn_tool.py` | Mod-11 doğrulama fonksiyonu (hazır, şu an çağrılmıyor) + TCKN maskeleme (`_mask` her yerde aktif) |
| `lookup_vehicle` | `tools/catalog_tool.py` | Marka/model → binek/ticari sınıflandırması (JSON katalog) |
| `retrieve_faq` | `tools/rag_tool.py` | ChromaDB'den semantic arama, chunk döndürme |
| `write_application` | `tools/db_tool.py` | Ön başvuruyu SQLite'a kayıt |
| `check_hgs_status` | `tools/hgs_tool.py` | Müşterinin aktif HGS'i var mı? (mock) |
| `register_hgs` | `tools/hgs_tool.py` | HGS başvurusu kayıt (mock) |

Tüm tool'lar `@tool` decorator ile LangChain'e kayıtlıdır. LLM hangi tool'u çağıracağına karar verir; iş mantığı tool içinde deterministik çalışır.

---

## 6. Güvenlik Katmanları

Bu projede uygulanan güvenlik katmanları:

| Katman | Bileşen | İşlev |
|---|---|---|
| **L4** | Input Guard (`guardrails/input_guard.py`) | PII maskeleme (TCKN, IBAN, kart), prompt injection tespiti, uzunluk limiti |
| **L5** | Orchestrator | Least-privilege tool erişimi, deterministik kural motoru |
| **L7** | Output Guard (`guardrails/output_guard.py`) | PII redaction, finansal taahhüt filtresi, FAQ kaynak zorunluluğu |

> L1 (TLS), L2 (IDP), L3 (API Gateway/WAF) bu projenin kapsam dışındaki altyapı gereksinimleridir.

Tüm guard ve tool olayları **SQLite Audit Log**'a yazılır (`audit_events` tablosu, append-only); TCKN gibi hassas veriler maskelenmiş haliyle kaydedilir.

---

## 7. Teknolojiler

### Çerçeveler ve Kütüphaneler

| Teknoloji | Kullanım Amacı |
|---|---|
| **LangGraph** | Multi-agent state machine, checkpointing, graph orchestration |
| **LangChain** | Tool dekoratörü, LLM soyutlama katmanı, mesaj formatları |
| **Gradio** | Chatbot web arayüzü |
| **OpenTelemetry** | Tracing altyapısı kurulu (`observability/tracer.py`), şu an ConsoleSpanExporter ile terminale yazıyor; Jaeger/Grafana bağlantısı yok |

### Yapay Zeka / LLM

| Teknoloji | Kullanım Amacı |
|---|---|
| **GPT-4o** (`ACTIVE_MODEL`) | Ana model — FAQ cevaplama, çok adımlı muhakeme |
| **GPT-4o-mini** (`ROUTER_MODEL`) | Hızlı model — niyet sınıflandırma, slot çıkarımı |
| **Claude (Anthropic)** | Opsiyonel alternatif model |
| **Gemini (Google)** | Opsiyonel alternatif model |
| **Jina Embeddings v3** | FAQ chunk embedding (API tabanlı) |

### Depolama

| Teknoloji | Kullanım Amacı |
|---|---|
| **ChromaDB** | Vektör veritabanı — FAQ RAG için semantic arama |
| **SQLite** | Ön başvuru ve HGS kayıtları |
| **LangGraph MemorySaver** | Oturum başına konuşma durumu (short-term memory) |

### Üretim Hedefi (Planlanan, Şu An Kodda Yok)

| Teknoloji | Kullanım Amacı |
|---|---|
| **vLLM** | LLM inference sunucusu (Llama 3.1 8B / 70B) |
| **BAAI/bge-m3** | On-prem embedding modeli |
| **Qdrant** | On-prem vektör veritabanı |
| **Redis** | Oturum store ve checkpointer |

---

## 8. Proje Yapısı

```
CarFinanceAgent/
└── chatbot/
    ├── app.py                      # Gradio arayüzü — giriş noktası
    ├── config.py                   # Merkezi konfigürasyon, iş kuralları, API key'ler
    ├── setup.py                    # Kurulum scripti (DB init, FAQ indeksleme)
    │
    ├── graph/
    │   ├── state.py                # ConversationState TypedDict
    │   └── builder.py              # LangGraph graph tanımı, edge'ler
    │
    ├── agents/
    │   ├── supervisor.py           # Niyet sınıflandırma, yönlendirme
    │   ├── intake_new.py           # Yeni araç slot filling
    │   ├── intake_used.py          # 2. el slot filling
    │   ├── faq_agent.py            # RAG tabanlı SSS yanıtlama
    │   ├── validation_agent.py     # TCKN, katalog, iş kuralı doğrulaması
    │   ├── submission_agent.py     # Başvuru kayıt
    │   └── crosssell_agent.py      # HGS çapraz satış
    │
    ├── tools/
    │   ├── tckn_tool.py            # TCKN mod-11 doğrulama
    │   ├── catalog_tool.py         # Araç kataloğu lookup
    │   ├── rag_tool.py             # ChromaDB retrieval
    │   ├── db_tool.py              # SQLite yazma/okuma
    │   └── hgs_tool.py             # HGS mock servisi
    │
    ├── guardrails/
    │   ├── input_guard.py          # L4: PII maskeleme, injection filtresi
    │   └── output_guard.py         # L7: PII redaction, finansal taahhüt filtresi
    │
    ├── memory/
    │   ├── checkpointer.py         # LangGraph MemorySaver konfigürasyonu
    │   └── vector_store.py         # ChromaDB bağlantısı, indexleme, arama
    │
    ├── llm/
    │   ├── router.py               # LLM seçimi (ACTIVE_MODEL / ROUTER_MODEL)
    │   └── prompts.py              # Tüm sistem prompt şablonları
    │
    ├── observability/
    │   ├── tracer.py               # OpenTelemetry tracing altyapısı (ConsoleSpanExporter)
    │   ├── audit_logger.py         # KVKK uyumlu SQLite audit log
    │   ├── metrics.py              # SQLite'tan metrik hesaplama (latency, dönüşüm oranı vb.)
    │   └── dashboard.py            # Gradio İzleme Paneli sekmesi
    │
    └── data/
        ├── faq.md                  # SSS kaynak dokümanı (Markdown)
        ├── vehicle_catalog.json    # Araç marka/model → binek/ticari kataloğu
        ├── chroma/                 # ChromaDB persist dizini
        └── applications.db         # SQLite veritabanı
```

---

## 9. İzleme Paneli

Uygulama, sohbet sekmesinin yanında **📊 İzleme Paneli** sekmesi içerir. Yenile butonuna basıldığında SQLite `audit_events` tablosundan anlık veri çeker; ek bir servis gerekmez.

### Gösterilen Metrikler

| Metrik | Kaynak |
|---|---|
| Toplam başvuru sayısı | `applications` tablosu |
| HGS dönüşüm oranı (%) | `hgs_registrations / applications` |
| Guard tetiklenme sayısı | `audit_events` — `event_type=guard_event` |
| LLM latency p50 / p95 (ms) | Son 100 `llm_call` olayından hesaplanır |

### Denetim Olay Tablosu

Son 30 audit olayı listelenir: oturum ID, saat, olay türü (`llm_call`, `tool_call`, `guard_event`, `slot_change`), agent adı ve özet.

### LLM Çağrı Tablosu

Son 50 LLM çağrısı; model, latency, token sayısı, prompt ve response önizlemesiyle gösterilir. Kısmî oturum UUID'siyle filtrelenebilir.

> İlgili dosyalar: `observability/dashboard.py` (Gradio sekme), `observability/metrics.py` (SQLite sorgular), `observability/audit_logger.py` (kayıt).

---

## 10. Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.11+
- OpenAI API key (GPT-4o için)
- Jina AI API key (FAQ embedding için)

### Adımlar

```bash
# 1. Sanal ortam oluştur ve aktifleştir
cd chatbot
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Ortam değişkenlerini ayarla
cp .env.example .env
# .env dosyasını düzenleyip API key'leri gir

# 4. İlk kurulumu çalıştır (DB init + FAQ indeksleme)
python setup.py

# 5. Uygulamayı başlat
python app.py
```

Uygulama başladıktan sonra `http://localhost:7860` adresinde Gradio arayüzü açılır.

---

## 11. Konfigürasyon

`config.py` dosyası merkezi konfigürasyon noktasıdır. Model değiştirmek için yalnızca `.env` dosyasını güncellemek yeterlidir:

```env
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
JINA_API_KEY=jina_...

# Model seçimi
ACTIVE_MODEL=gpt-4o           # veya: claude-opus-4-7, gemini-2.0-flash
ROUTER_MODEL=gpt-4o-mini      # hızlı sınıflandırma modeli
```

### İş Kurallarını Değiştirme

Tüm eşikler `config.py`'de sabitler olarak tanımlıdır:

```python
NEW_CAR_MAX_INVOICE = 7_000_000.0       # TL
NEW_CAR_FINANCE_RATIO = 0.60            # %60
NEW_CAR_GUARANTOR_THRESHOLD = 5_000_000.0
USED_CAR_MAX_AGE = 5                    # yıl
USED_CAR_FINANCE_RATIO = 0.40           # %40
USED_CAR_MAX_FINANCE = 3_000_000.0      # TL
```

---

## 12. Diyagramlar

Detaylı sistem mimarisi, use case akışları ve güvenlik katmanı diyagramları için [chatbot/DIAGRAMS.md](chatbot/DIAGRAMS.md) dosyasına bakın.

> **Not:** Diyagramlar Mermaid formatındadır. VS Code'da görüntülemek için **"Markdown Preview Mermaid Support"** uzantısını yükleyin, ardından `Ctrl+Shift+V` ile önizleyin. Alternatif olarak [mermaid.live](https://mermaid.live) sitesini kullanabilirsiniz.

### Diyagram Listesi

| # | Diyagram | Açıklama |
|---|---|---|
| 1 | Agentic Tasarım | Agent'lar, tool'lar ve shared memory |
| 2 | Ana Akış (Use Case) | Yeni araç ve 2. el dalları, SSS entegrasyonu |
| 3 | Hedefli Güncelleme Akışı | Onay ekranında alan değiştirme mantığı |
| 4 | Paralel FAQ Akışı | Sequence diagram — snapshot / restore döngüsü |
