"""Preview the local cases, or upload them to the Langfuse dataset."""

import argparse
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parent
DATASET_NAME = "invoice-agent-poc"


def dataset_items():
    cases = json.loads((ROOT / "data/cases.json").read_text())
    references = json.loads((ROOT / "data/expected_outputs.json").read_text())
    case_ids = [case["id"] for case in cases]
    reference_ids = [reference["id"] for reference in references]
    if (len(set(case_ids)) != len(case_ids)
            or len(set(reference_ids)) != len(reference_ids)
            or set(case_ids) != set(reference_ids)):
        raise ValueError("Inputs and expected outputs must have matching unique IDs")
    expected = {reference["id"]: reference["expected"] for reference in references}
    return [{
        # Stable, dataset-specific UUIDs make repeat uploads update existing items.
        "id": str(uuid5(NAMESPACE_URL, DATASET_NAME + "/" + case["id"])),
        "input": {"invoice_text": case["input"]},
        "expected_output": expected[case["id"]],
        "metadata": {"case_id": case["id"], "category": case["category"]},
    } for case in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="Upload the reviewed payload")
    args = parser.parse_args()
    items = dataset_items()
    if not args.upload:
        print(json.dumps({"dataset": DATASET_NAME, "items": items}, indent=2))
        return
    from observability import Tracer
    tracer = Tracer(True)
    try:
        tracer.client.create_dataset(name=DATASET_NAME,
                                     description="30 synthetic invoice evaluation cases")
        for item in items:
            tracer.client.create_dataset_item(dataset_name=DATASET_NAME, **item)
        print(f"Uploaded {len(items)} items to {DATASET_NAME}")
    finally:
        tracer.flush()


if __name__ == "__main__":
    main()
