from __future__ import annotations

import json
from typing import Any, Dict

try:
	from crewai import Task
except Exception:  # pragma: no cover - allow import without crewai for offline
	Task = object  # type: ignore

from .tools import parse_wearable_json, lookup_guidelines, apply_treatment_safety


def create_symptom_collection_task(agent, patient_context: Dict[str, Any]) -> Task:
	"""
	patient_context should include keys: demographics, chief_complaint, symptoms (list), history, meds, allergies, wearable (dict or str)
	"""
	wearable_summary = parse_wearable_json(patient_context.get("wearable"))
	instructions = f"""
You will produce a structured JSON summary. Use exactly these keys:
- demographics
- chief_complaint
- symptoms
- history
- current_medications
- allergies
- wearable_summary

Wearable summary:
{json.dumps(wearable_summary.__dict__, indent=2)}

Patient context:
{json.dumps(patient_context, indent=2)}

Return only JSON.
"""
	return Task(description=instructions, agent=agent, output_json=True)


def create_diagnosis_task(agent, collected: Dict[str, Any]) -> Task:
	guidelines = lookup_guidelines([*(collected.get("symptoms") or []), collected.get("chief_complaint", "")])
	instructions = f"""
Using the structured collection and minimal guideline data, produce a differential diagnosis.
Return JSON with keys:
- differentials: list of objects {{name, likelihood_percent, rationale}}
- red_flags: list of strings
- suggested_workup: list of strings
- notes: short string

Input structured collection:
{json.dumps(collected, indent=2)}

Guideline snippets:
{json.dumps(guidelines, indent=2)}

Return only JSON.
"""
	return Task(description=instructions, agent=agent, output_json=True)


def create_treatment_task(agent, collected: Dict[str, Any], diagnosis: Dict[str, Any]) -> Task:
	instructions = f"""
Given the structured collection and differential, propose an initial plan with safety in mind.
Steps:
1) For top 1-3 likely conditions, propose first-line options (medications or lifestyle) as JSON objects {{name, dose, frequency, notes}}.
2) Include non-pharmacologic recommendations where applicable.
3) Include follow_up: list of strings.
4) Do not recommend items that conflict with allergies or duplicate current meds.

Return JSON with keys:
- plan: list of recommendation objects
- follow_up: list of strings
- rationale: short string

Collection:
{json.dumps(collected, indent=2)}

Differential:
{json.dumps(diagnosis, indent=2)}

Return only JSON.
"""
	return Task(description=instructions, agent=agent, output_json=True)