"""Run the real SDK loop with a scripted model; no OpenAI calls are made."""

import json
import unittest
from pathlib import Path
from agents import Model, ModelResponse, Usage, MaxTurnsExceeded
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from agent import run_agent
from agent import calculate_total, Invoice, validate_invoice
from evaluate import equal, score

CASES = json.loads(
    (Path(__file__).resolve().parents[1] / "data/cases.json").read_text()
)

EXPECTED_OUTPUTS = {
    item["id"]: item["expected"]
    for item in json.loads(
        (Path(__file__).resolve().parents[1] / "data/expected_outputs.json").read_text()
    )
}
# Offline fixtures use reference fields to script replies; live agents never see them.
CASES = [{**case, "expected": EXPECTED_OUTPUTS[case["id"]]} for case in CASES]


class ScriptedModel(Model):
    def __init__(self, invoice, calls=None):
        self.calls = calls or [
            ("calculate_total", {"line_items": invoice["line_items"]}),
            ("final", {"invoice": invoice}),
        ]
        self.requests = []

    async def get_response(self, system_instructions, input, *args, **kwargs):
        self.requests.append(input)
        name, arguments = self.calls[min(len(self.requests) - 1, len(self.calls) - 1)]
        if name == "final":
            output = ResponseOutputMessage(
                type="message", id="msg_final", role="assistant", status="completed",
                content=[ResponseOutputText(
                    type="output_text", text=json.dumps(arguments["invoice"]), annotations=[],
                )],
            )
        else:
            output = ResponseFunctionToolCall(
                type="function_call", id=f"fc_{len(self.requests)}",
                call_id=f"call_{len(self.requests)}", name=name,
                arguments=json.dumps(arguments),
            )
        return ModelResponse(
            output=[output], usage=Usage(requests=1, input_tokens=20, output_tokens=10),
            response_id=None,
        )

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("This POC does not stream")
        yield


class InvoiceTests(unittest.TestCase):
    def test_all_agent_paths(self):
        for case in CASES:
            with self.subTest(case=case["id"]):
                model = ScriptedModel(case["expected"]["invoice"])
                result = run_agent(case["input"], model=model)
                self.assertEqual(result["issues"], case["expected"]["issues"])
                self.assertEqual(score(result, case["expected"])[0]["task_success"], 1)
                self.assertEqual(
                    model.requests[0], [{"role": "user", "content": case["input"]}]
                )
                self.assertEqual(result["token_usage"], {"input": 40, "output": 20})

    def test_decimal_rounding(self):
        items = [{"description": "tiny", "quantity": "3", "unit_price": "0.10"}]
        self.assertEqual(calculate_total(items)["calculated_total"], "0.30")
        items[0]["unit_price"] = "0.105"
        self.assertEqual(calculate_total(items)["calculated_total"], "0.32")

    def test_invalid_money(self):
        for amount in ("NaN", "Infinity", "-1"):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                calculate_total(
                    [{"description": "x", "quantity": "1", "unit_price": amount}]
                )

    def test_fabricated_calculation_rejected(self):
        with self.assertRaises(ValueError):
            validate_invoice(
                Invoice.model_validate(CASES[0]["expected"]["invoice"]),
                {"calculated_total": "99"},
            )

    def test_missing_calculator_rejected(self):
        invoice = CASES[0]["expected"]["invoice"]
        model = ScriptedModel(invoice, [("final", {"invoice": invoice})])
        with self.assertRaisesRegex(ValueError, "must calculate"):
            run_agent(CASES[0]["input"], model=model)

    def test_repeated_tool_calls_are_bounded(self):
        invoice = CASES[0]["expected"]["invoice"]
        model = ScriptedModel(invoice, [("calculate_total", {"line_items": invoice["line_items"]})])
        with self.assertRaises(MaxTurnsExceeded):
            run_agent(CASES[0]["input"], model=model, max_steps=2)

    def test_scorer_detects_wrong_fields_and_flags(self):
        result = run_agent(
            CASES[0]["input"], model=ScriptedModel(CASES[0]["expected"]["invoice"])
        )
        result["invoice"]["vendor"] = "Invented"
        result["issues"] = ["total_mismatch"]
        scores, _ = score(result, CASES[0]["expected"])
        self.assertLess(scores["field_accuracy"], 1)
        self.assertEqual(scores["issues_exact"], 0)
        self.assertEqual(scores["task_success"], 0)

    def test_numeric_comparison_preserves_text_fields(self):
        self.assertFalse(equal("1", "001"))
        self.assertFalse(equal("1", "1.00"))
        self.assertTrue(equal("26", "26.00", numeric=True))
        self.assertTrue(equal(
            {"quantity": "2.0", "unit_price": "10"},
            {"quantity": "2", "unit_price": "10.00"},
        ))
        self.assertFalse(equal(
            {"description": "001"}, {"description": "1"},
        ))

    def test_final_invoice_must_match_calculated_items(self):
        original = CASES[0]["expected"]["invoice"]
        changed = json.loads(json.dumps(original))
        changed["line_items"][0].update(quantity="1", unit_price="20.00")
        model = ScriptedModel(
            original,
            [
                ("calculate_total", {"line_items": original["line_items"]}),
                ("final", {"invoice": changed}),
            ],
        )
        with self.assertRaisesRegex(ValueError, "must calculate"):
            run_agent(CASES[0]["input"], model=model, max_steps=2)

    def test_provider_failure_propagates(self):
        class FailingModel(ScriptedModel):
            async def get_response(self, *args, **kwargs):
                raise RuntimeError("provider failure")

        with self.assertRaisesRegex(RuntimeError, "provider failure"):
            run_agent(
                CASES[0]["input"], model=FailingModel(CASES[0]["expected"]["invoice"])
            )


if __name__ == "__main__":
    unittest.main()
