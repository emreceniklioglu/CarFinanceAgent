"""
observability/tracer.py — OpenTelemetry Distributed Tracing

REPORT Karşılığı: "Audit Log + Trace" (OpenTelemetry ile distributed tracing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Her konuşma turu bir TRACE; her agent çağrısı bir SPAN olarak kaydedilir.

Span hiyerarşisi:
  conversation_turn (root span)
  ├── input_guard
  ├── supervisor
  │   └── llm_call
  ├── intake_agent / faq_agent / validation_agent
  │   ├── llm_call
  │   └── tool_call
  └── output_guard

Geliştirmede: ConsoleSpanExporter ile terminale yazdırılır.
Üretimde: OTLP exporter ile Jaeger/Grafana Tempo'ya gönderilir.
"""
from contextlib import contextmanager
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

# Uygulama kimliği
resource = Resource.create({"service.name": "arac-finansman-chatbot", "service.version": "1.0.0"})

# Provider kurulumu (uygulama başında bir kez yapılır)
_provider = TracerProvider(resource=resource)
_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(_provider)

tracer = trace.get_tracer("chatbot.tracer")


@contextmanager
def trace_span(name: str, attributes: dict = None):
    """
    Context manager olarak span başlatır.

    Kullanım:
        with trace_span("validation_agent", {"session_id": "abc"}):
            # bu blok içindeki işlemler span içinde yer alır
            ...
    """
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise


def get_current_trace_id() -> str:
    """Aktif trace ID'sini döndürür (log korelasyonu için)."""
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return "no-trace"
