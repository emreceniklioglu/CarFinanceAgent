"""
llm/factory.py — LLM Model Switcher (LiteLLM Wrapper)

AGENTIC PATTERN: Unified LLM Interface
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LiteLLM, OpenAI / Anthropic / Google API'lerini TEK bir interface
uzerinden kullanmana izin verir.

Model degistirmek icin config.py'deki ACTIVE_MODEL'i guncelle:
  "gpt-4o"                         -> OpenAI GPT-4o
  "claude-sonnet-4-6"              -> Anthropic Claude Sonnet
  "gemini/gemini-flash-latest"     -> Google Gemini Flash

Iki katmanli mimari (REPORT Q6):
  - ROUTER (kucuk/hizli): intent siniflandirma, slot cikarimi
  - MAIN   (buyuk/guclu): FAQ cevaplama, karmasik muhakeme
"""
import time
import litellm
from litellm.exceptions import (
    ServiceUnavailableError,
    RateLimitError,
    Timeout,
    InternalServerError,
    APIConnectionError,
)
import config

# LiteLLM verbose loglamasini kapat
litellm.set_verbose = False

# Gecici (transient) hatalarda yeniden dene
_RETRYABLE_ERRORS = (
    ServiceUnavailableError,  # 503 - model yogun
    RateLimitError,           # 429
    Timeout,
    InternalServerError,      # 500
    APIConnectionError,
)
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # saniye


class LLMResponse:
    """LLM cevabini tasimak icin minimal wrapper."""
    def __init__(self, content: str):
        self.content = content


class LiteLLMWrapper:
    """
    litellm.completion() uzerinde LangChain benzeri invoke() arayuzu.
    Tum agent'lar llm.invoke(messages) -> response.content seklinde kullanir.
    """

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def invoke(self, messages: list) -> LLMResponse:
        """
        messages: dict listesi [{"role": "system", "content": "..."}]
                  veya LangChain BaseMessage listesi
        """
        formatted = _format_messages(messages)
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=formatted,
                    temperature=self.temperature,
                )
                choice = response.choices[0]
                content = choice.message.content
                # Gemini bazen güvenlik/RECITATION filtresinde None içerik döndürür.
                # Çağrı noktaları .split() vb. çağırdığı için None'ı boş string'e çevir.
                if content is None:
                    finish = getattr(choice, "finish_reason", "unknown")
                    print(f"[WARN llm] {self.model} boş içerik döndü (finish_reason={finish})", flush=True)
                    content = ""
                return LLMResponse(content=content)
            except _RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BASE_DELAY * (2 ** attempt))  # 1s, 2s, 4s
        raise last_error


def _format_messages(messages: list) -> list[dict]:
    """LangChain mesaj nesnelerini ya da dict'leri litellm formatina donustur."""
    result = []
    for m in messages:
        if isinstance(m, dict):
            result.append(m)
        elif hasattr(m, "type"):
            # LangChain BaseMessage: HumanMessage, AIMessage, SystemMessage
            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            role = role_map.get(m.type, "user")
            result.append({"role": role, "content": m.content})
        else:
            result.append({"role": "user", "content": str(m)})
    return result


def get_llm(size: str = "large", temperature: float = 0.0) -> LiteLLMWrapper:
    """
    size="large" -> Ana model (ACTIVE_MODEL)
    size="small" -> Router model (ROUTER_MODEL)

    config modulu direkt okunur -> UI'dan model degisikliginde aninda yansir.
    """
    model_name = config.ACTIVE_MODEL if size == "large" else config.ROUTER_MODEL
    return LiteLLMWrapper(model=model_name, temperature=temperature)


def get_router_llm() -> LiteLLMWrapper:
    """Kisayol: intent ve slot gorevleri icin kucuk hizli model."""
    return get_llm(size="small")


def get_main_llm() -> LiteLLMWrapper:
    """Kisayol: FAQ ve karmasik gorevler icin buyuk guclu model."""
    return get_llm(size="large")
