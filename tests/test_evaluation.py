"""Evaluation checks and judge failures without calling either cloud service."""

import copy
import asyncio
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluate import (Groundedness, deterministic_evaluator, groundedness_evaluator,
                      judge_groundedness, score, summarize)
from setup_dataset import dataset_items
from test_invoice_agent import CASES, ScriptedModel
from agent import run_agent
from langfuse import Langfuse


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = CASES[0]
        self.result = run_agent(self.case["input"],
                                model=ScriptedModel(self.case["expected"]["invoice"]))

    def test_tool_arguments_are_checked_against_reference(self):
        wrong = copy.deepcopy(self.result)
        # Same total, different quantities/prices: calling the tool is insufficient.
        wrong["trajectory"][0]["input"]["line_items"][0].update(
            quantity="1", unit_price="20.00")
        scores, _ = score(wrong, self.case["expected"])
        self.assertEqual(scores["calculator_used"], 1)
        self.assertEqual(scores["calculator_arguments_correct"], 0)
        self.assertEqual(scores["task_success"], 0)
        wrong["trajectory"] = []
        self.assertEqual(score(wrong, self.case["expected"])[0]["calculator_arguments_correct"], 0)

    def test_structured_judge_pass_and_fail(self):
        for value in (0, 1):
            verdict = Groundedness(groundedness=value, explanation="Vendor is supported" if value else "Vendor was invented")
            parse = SimpleNamespace(parse=lambda **kwargs: SimpleNamespace(output_parsed=verdict))
            client = SimpleNamespace(responses=parse)
            self.assertEqual(judge_groundedness(self.case["input"], self.result,
                                                client=client), verdict)

    def test_judge_failures_are_errors_not_zero_scores(self):
        for problem in (ValueError("malformed output"), RuntimeError("provider failed")):
            with patch("evaluate.judge_groundedness", side_effect=problem):
                evaluation = groundedness_evaluator(
                    input={"invoice_text": self.case["input"]}, output=self.result)
            self.assertEqual(evaluation.name, "evaluator_error")
            self.assertEqual(evaluation.value, type(problem).__name__)
        client = SimpleNamespace(responses=SimpleNamespace(
            parse=lambda **kwargs: SimpleNamespace(output_parsed=None)))
        with self.assertRaises(ValueError):
            judge_groundedness(self.case["input"], self.result, client=client)
        client.responses.parse = lambda **kwargs: SimpleNamespace(
            output_parsed={"groundedness": 2, "explanation": "bad schema"})
        with self.assertRaises(ValueError):
            judge_groundedness(self.case["input"], self.result, client=client)

    def test_judge_score_has_explanation_and_is_separate(self):
        with patch("evaluate.judge_groundedness", return_value=Groundedness(
                groundedness=0, explanation="Unsupported vendor")):
            evaluation = groundedness_evaluator(input={"invoice_text": "text"}, output=self.result)
        self.assertEqual(evaluation.name, "groundedness")
        self.assertEqual(evaluation.value, 0)
        self.assertEqual(evaluation.comment, "Unsupported vendor")
        self.assertEqual(score(self.result, self.case["expected"])[0]["task_success"], 1)

    def test_summary_keeps_errors_and_coverage_visible(self):
        evaluations = deterministic_evaluator(output=self.result, expected_output=self.case["expected"])
        with patch("evaluate.judge_groundedness", side_effect=ValueError):
            evaluations.append(groundedness_evaluator(input={}, output=self.result))
        items = [SimpleNamespace(output=self.result, evaluations=evaluations),
                 SimpleNamespace(output={"execution_error": "RuntimeError"}, evaluations=[])]
        summary = {e.name: e for e in summarize(item_results=items)}
        self.assertEqual(summary["deterministic_pass_rate"].value, 0.5)
        self.assertEqual(summary["mean_task_success"].value, 1)
        self.assertEqual(summary["execution_errors"].value, 1)
        self.assertEqual(summary["evaluator_errors"].value, 1)
        self.assertNotIn("mean_groundedness", summary)
        self.assertEqual(deterministic_evaluator(output=items[1].output, expected_output={}), [])

    def test_dataset_payload_is_stable_and_matches_local_cases(self):
        items = dataset_items()
        self.assertEqual(items, dataset_items())
        self.assertEqual(len(items), 30)
        self.assertEqual(len({item["id"] for item in items}), 30)
        for case, item in zip(CASES, items):
            self.assertEqual(item["input"], {"invoice_text": case["input"]})
            self.assertEqual(item["expected_output"], case["expected"])
            self.assertNotIn("review_status", item["expected_output"])

    def test_dataset_coverage_and_source_answers(self):
        self.assertEqual(Counter(case["category"] for case in CASES), {
            "valid": 8, "wrong_total": 5, "missing_fields": 5,
            "decimal_rounding": 4, "ambiguous": 4, "embedded_instructions": 4})
        self.assertEqual(Counter(case["expected"]["status"] for case in CASES),
                         {"valid": 14, "needs_review": 16})
        # Read only the labelled source fields/items, excluding adversarial footers.
        # This catches a reference answer accidentally disagreeing with its input.
        for case in CASES:
            with self.subTest(case=case["id"]):
                lines = case["input"].split("\n\n")[0].splitlines()
                source = dict(line.split(": ", 1) for line in lines if ": " in line)
                invoice = case["expected"]["invoice"]
                for field, label in (("vendor", "Vendor"), ("invoice_number", "Invoice number"),
                                     ("total", "Total")):
                    self.assertEqual(invoice[field], source.get(label))
                source_date = source.get("Date")
                if source_date == "October 9, 2026":
                    source_date = "2026-10-09"
                self.assertEqual(invoice["date"], source_date)
                currency = source.get("Currency")
                if currency in {"$", "kr"}:
                    currency = None
                self.assertEqual(invoice["currency"], currency)
                source_items = []
                for line in lines:
                    if " | quantity " in line:
                        description, quantity, price = line.split(" | ")
                        source_items.append({"description": description,
                                             "quantity": quantity.removeprefix("quantity "),
                                             "unit_price": price.removeprefix("unit price ")})
                self.assertEqual(invoice["line_items"], source_items)
                total = sum((Decimal(item["quantity"]) * Decimal(item["unit_price"])).quantize(
                    Decimal(".01"), rounding=ROUND_HALF_UP) for item in source_items)
                self.assertEqual(Decimal(case["expected"]["calculated_total"]), total)

    def test_sdk_experiment_runs_all_three_layers(self):
        # Exercise the real experiment runner, but no network or real model calls.
        client = Langfuse(public_key="pk-lf-offline-experiment", tracing_enabled=False)
        previous_loop = asyncio.get_event_loop()

        def execute(item):
            with asyncio.Runner() as runner:
                asyncio.set_event_loop(runner.get_loop())
                return run_agent(item["input"]["invoice_text"],
                                 model=ScriptedModel(item["expected_output"]["invoice"]))

        async def task(*, item, **kwargs):
            return await asyncio.to_thread(execute, item)

        try:
            with patch("evaluate.judge_groundedness", return_value=Groundedness(
                    groundedness=1, explanation="Scripted offline verdict")):
                result = client.run_experiment(
                    name="offline-integration", data=dataset_items(), task=task,
                    evaluators=[deterministic_evaluator, groundedness_evaluator],
                    run_evaluators=[summarize], max_concurrency=1)
            self.assertEqual(len(result.item_results), 30)
            self.assertTrue(all(len(item.evaluations) == 8 for item in result.item_results))
            summary = {evaluation.name: evaluation.value for evaluation in result.run_evaluations}
            self.assertEqual(summary["deterministic_pass_rate"], 1)
            self.assertEqual(summary["execution_errors"], 0)
            self.assertEqual(summary["evaluator_errors"], 0)
        finally:
            client.shutdown()
            asyncio.set_event_loop(previous_loop)


if __name__ == "__main__":
    unittest.main()
