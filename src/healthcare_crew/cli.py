from __future__ import annotations

import json
from typing import Optional

import click

from .crew import run_pipeline


@click.command()
@click.option("--patient", required=True, help="Patient symptom narrative (comma-separated symptoms ok)")
@click.option("--history", required=False, default=None, help="Medical history items (comma-separated)")
@click.option("--wearable-json", required=False, default=None, help="JSON string of wearable data")
def main(patient: str, history: Optional[str], wearable_json: Optional[str]):
	wearable = json.loads(wearable_json) if wearable_json else None
	outputs = run_pipeline(patient_text=patient, medical_history_text=history, wearable=wearable)
	click.echo(json.dumps(outputs, indent=2))


if __name__ == "__main__":
	main()