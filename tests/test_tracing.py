"""Verify SDK agent/tool spans using an in-memory exporter, without remote calls."""

import unittest
from unittest.mock import Mock
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import agent
from agent import calculator_tool
from observability import Tracer, run_traced_agent
from test_invoice_agent import CASES, ScriptedModel


class TracingTests(unittest.TestCase):
    def test_sdk_agent_and_tool_observations(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        tracer = Tracer()
        tracer.client = Langfuse(
            public_key="pk-lf-offline-sdk",
            secret_key="sk-lf-offline-sdk",
            tracer_provider=provider,
            span_exporter=exporter,
        )
        tracer.url = Mock(return_value=None)
        original = calculator_tool.on_invoke_tool
        original_validation = agent.validate_invoice
        tracer.instrument_tool()
        tracer.instrument_validation()
        try:
            result = run_traced_agent(
                CASES[0]["input"],
                model=ScriptedModel(CASES[0]["expected"]["invoice"]),
                tracer=tracer,
            )
            tracer.flush()
            spans = exporter.get_finished_spans()
            self.assertEqual([s.name for s in spans].count("invoice-request"), 1)
            self.assertEqual(len(spans), 3)
            validation = next(s for s in spans if s.name == "validate_invoice")
            parent = next(s for s in spans if s.name == "invoice-request")
            self.assertEqual(validation.parent.span_id, parent.context.span_id)
            self.assertIn("valid", str(validation.attributes))
            self.assertTrue(any(s.name == "calculate_total" for s in spans))
            self.assertEqual(len({s.context.trace_id for s in spans}), 1)
            self.assertTrue(result["trace_id"])
            self.assertTrue(any(s.name == "invoice-request" for s in spans))
            self.assertTrue(any("26.00" in str(s.attributes) for s in spans))
        finally:
            calculator_tool.on_invoke_tool = original
            agent.validate_invoice = original_validation
            tracer.client.shutdown()


if __name__ == "__main__":
    unittest.main()
