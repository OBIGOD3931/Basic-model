from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich import print
from dotenv import load_dotenv

from .crew import run_healthcare_crew
from .llm_config import have_any_api_key

app = typer.Typer(help="Healthcare Diagnostic Support System (CrewAI)")


def _load_json(path: Optional[Path]) -> Optional[dict]:
	if not path:
		return None
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


@app.command()
def run(
	patient_file: Path = typer.Option(..., help="Path to patient JSON"),
	wearable_file: Optional[Path] = typer.Option(None, help="Path to wearable JSON"),
	dry_run: bool = typer.Option(False, help="Avoid network LLM calls; use offline rules"),
):
	"""Run the crew for the given patient JSON and optional wearable JSON."""
	load_dotenv()
	if dry_run:
		os.environ["OFFLINE_MODE"] = "true"
	patient = _load_json(patient_file) or {}
	if wearable_file:
		patient["wearable"] = _load_json(wearable_file)

	if dry_run:
		print("[yellow]Dry run enabled (OFFLINE_MODE).[/yellow]")
	elif not have_any_api_key():
		print("[yellow]No API key detected. Use --dry-run or set an API key.[/yellow]")

	result = run_healthcare_crew(patient)
	print_json(result)


@app.command()
def demo(dry_run: bool = typer.Option(False, help="Avoid network LLM calls; use offline rules")):
	"""Run the crew with bundled sample data."""
	load_dotenv()
	if dry_run:
		os.environ["OFFLINE_MODE"] = "true"
	project_root = Path(__file__).resolve().parent.parent.parent
	patient_path = project_root / "data" / "sample_patient.json"
	wearable_path = project_root / "data" / "sample_wearable.json"

	if dry_run:
		print("[yellow]Dry run enabled (OFFLINE_MODE).[/yellow]")
	elif not have_any_api_key():
		print("[yellow]No API key detected. Use --dry-run or set an API key.[/yellow]")

	patient = _load_json(patient_path) or {}
	patient["wearable"] = _load_json(wearable_path)
	result = run_healthcare_crew(patient)
	print_json(result)


def print_json(obj):
	print("[bold]\n==== Result ====\n[/bold]")
	print(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	app()