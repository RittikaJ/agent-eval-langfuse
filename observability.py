"""Keep optional Langfuse tracing out of the invoice business logic."""

import os
import json
import asyncio
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


class NullObservation:
    trace_id = None

    def update(self, **kwargs):
        pass


class Tracer:
    def __init__(self, enabled=False):
        self.client = None
        if enabled:
            required = (
                "LANGFUSE_PUBLIC_KEY",
                "LANGFUSE_SECRET_KEY",
                "LANGFUSE_BASE_URL",
            )
            if not all(os.environ.get(k) for k in required):
                raise ValueError(
                    "Configure Langfuse credentials in .env before using --trace"
                )
            from langfuse import get_client

            self.client = get_client()
            # Trace model calls once with Langfuse's OpenAI integration.
            # SDK workflow/turn containers add no useful detail to this small agent.
            self.instrument_tool()
            self.instrument_validation()

    def instrument_tool(self):
        """Record calculator arguments and result, excluding the Python context."""
        from agent import calculator_tool
        original = calculator_tool.on_invoke_tool
        if getattr(original, "_invoice_traced", False):
            return

        async def traced_tool(context, arguments):
            with self.observation("calculate_total", kind="tool",
                                  input=json.loads(arguments)) as observation:
                output = await original(context, arguments)
                observation.update(output=json.loads(output))
                return output

        traced_tool._invoice_traced = True
        calculator_tool.on_invoke_tool = traced_tool

    def instrument_validation(self):
        """Show Python validation separately from model and tool calls."""
        import agent
        original = agent.validate_invoice
        if getattr(original, "_invoice_traced", False):
            return

        def traced_validation(invoice, calculation):
            with self.observation(
                "validate_invoice",
                input={"invoice": invoice.model_dump(mode="json"),
                       "calculation": calculation},
            ) as observation:
                result = original(invoice, calculation)
                observation.update(output={
                    key: result[key] for key in ("total_matches", "issues", "status")
                })
                return result

        traced_validation._invoice_traced = True
        agent.validate_invoice = traced_validation

    @contextmanager
    def observation(self, name, kind="span", **kwargs):
        if self.client is None:
            yield NullObservation()
        else:
            with self.client.start_as_current_observation(
                name=name, as_type=kind, **kwargs
            ) as obs:
                yield obs

    @contextmanager
    def attributes(self, metadata):
        if self.client is None:
            yield
        else:
            from langfuse import propagate_attributes

            with propagate_attributes(tags=["invoice-poc"], metadata=metadata):
                yield

    def flush(self):
        if self.client:
            self.client.flush()

    def url(self, trace_id):
        if not self.client or not trace_id:
            return None
        try:
            return self.client.get_trace_url(trace_id=trace_id)
        except Exception:
            # A dashboard URL lookup must not discard a successful invoice.
            return None


def run_traced_agent(text, *, tracer, case_id="interactive", model=None, max_steps=6):
    """Wrap the invoice agent with optional tracing and attach its dashboard link."""
    from agents import RunConfig, OpenAIResponsesModel
    from agent import run_agent

    client = None
    if tracer.client is not None and model is None:
        from langfuse.openai import AsyncOpenAI
        client = AsyncOpenAI(timeout=30, max_retries=0)
        create_response = client.responses.create

        async def named_response(**kwargs):
            # Name the step from its context, rather than assuming a turn number.
            has_tool_result = any(
                isinstance(item, dict) and item.get("type") == "function_call_output"
                for item in kwargs.get("input", [])
            )
            kwargs["name"] = "return_invoice" if has_tool_result else "extract_invoice"
            return await create_response(**kwargs)

        client.responses.create = named_response
        model = OpenAIResponsesModel(os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), client)

    with tracer.attributes(
        {"case_id": case_id}
    ), tracer.observation(
        "invoice-request", kind="agent", input={"invoice_text": text}
    ) as observation:
        try:
            result = run_agent(
                text,
                model=model,
                max_steps=max_steps,
                run_config=RunConfig(
                    tracing_disabled=True,
                ),
            )
        finally:
            if client is not None:
                asyncio.get_event_loop().run_until_complete(client.close())
        result["trace_id"] = observation.trace_id
        result["trace_url"] = tracer.url(observation.trace_id)
        observation.update(output=result)
        return result
