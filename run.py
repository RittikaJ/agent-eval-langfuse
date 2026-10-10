"""Run selected invoices, score their results, and save them locally."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from evaluate import score
from observability import Tracer, run_traced_agent

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Run the real invoice agent and save evaluation artifacts."
    )
    parser.add_argument(
        "--case",
        default="invoice-001",
        help="Case ID, or all to run all local cases",
    )
    parser.add_argument(
        "--trace", action="store_true", help="Send observations and scores to Langfuse"
    )
    args = parser.parse_args()
    cases = json.loads((ROOT / "data/cases.json").read_text())
    expected_outputs = {
        item["id"]: item["expected"]
        for item in json.loads((ROOT / "data/expected_outputs.json").read_text())
    }
    selected = (
        cases if args.case == "all" else [c for c in cases if c["id"] == args.case]
    )
    if not selected:
        parser.error("Unknown case ID")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error(
            "Set OPENAI_API_KEY in invoice-agent-poc/.env before running the real agent"
        )
    if any(case["id"] not in expected_outputs for case in selected):
        parser.error("A selected case has no expected output")
    tracer = Tracer(args.trace)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results" / run_id
    destination.mkdir(parents=True)
    failed = False
    try:
        with (destination / "runs.jsonl").open("w") as artifact:
            for case in selected:
                started = time.monotonic()
                record = {"case_id": case["id"], "run_id": run_id, "mode": "real-model"}
                obs = None
                try:
                    with tracer.attributes({"case_id": case["id"]}):
                        result = run_traced_agent(
                            case["input"], tracer=tracer, case_id=case["id"]
                        )
                        scores, fields = score(result, expected_outputs[case["id"]])
                        if tracer.client:
                            for name, value in scores.items():
                                tracer.client.create_score(
                                    name=name,
                                    value=value,
                                    trace_id=result["trace_id"],
                                    data_type="NUMERIC",
                                )
                        record.update(result=result, scores=scores, field_checks=fields)
                    print(case["id"], result["status"], scores)
                    if result["trace_url"]:
                        print("Trace:", result["trace_url"])
                except Exception as exc:
                    failed = True
                    record["error"] = {
                        "type": type(exc).__name__,
                        "message": "Run failed; inspect local configuration or the trace.",
                    }
                    body = getattr(exc, "body", None)
                    if isinstance(body, dict):
                        record["error"]["provider_code"] = body.get("code")
                    record["trace_id"] = obs.trace_id if obs else None
                    record["trace_url"] = tracer.url(record["trace_id"])
                    print(
                        case["id"],
                        "FAILED",
                        type(exc).__name__,
                        record["error"].get("provider_code"),
                    )
                    if record.get("trace_url"):
                        print("Trace:", record["trace_url"])
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                artifact.write(json.dumps(record) + "\n")
                artifact.flush()
    finally:
        tracer.flush()
    print("Artifacts:", destination)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
