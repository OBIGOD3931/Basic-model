from __future__ import annotations

import os
from typing import Any, Dict

try:
	from crewai import Crew, Process
	export_available = True
except Exception:  # pragma: no cover - allow offline/demo
	export_available = False
	Crew = object  # type: ignore
	Process = object  # type: ignore

from .agents import (
	build_symptom_collector_agent,
	build_diagnosis_agent,
	build_treatment_planner_agent,
)
from .tools import parse_wearable_json, lookup_guidelines, apply_treatment_safety


def _offline_flow(patient_context: Dict[str, Any]) -> Dict[str, Any]:
	# Deterministic offline mock: bypass LLM and use simple rules
	wearable = parse_wearable_json(patient_context.get("wearable"))
	collected = {
		"demographics": patient_context.get("demographics"),
		"chief_complaint": patient_context.get("chief_complaint"),
		"symptoms": patient_context.get("symptoms"),
		"history": patient_context.get("history"),
		"current_medications": patient_context.get("current_medications"),
		"allergies": patient_context.get("allergies"),
		"wearable_summary": wearable.__dict__,
	}
	guides = lookup_guidelines([*(collected.get("symptoms") or []), collected.get("chief_complaint", "")])
	# naive diff
	differentials = []
	if "chest_pain" in (" ".join(collected.get("symptoms") or [])).lower() or "chest pain" in (collected.get("chief_complaint") or "").lower():
		differentials.append({"name": "acute coronary syndrome", "likelihood_percent": 35, "rationale": "risk factors + chest pain"})
		differentials.append({"name": "musculoskeletal", "likelihood_percent": 25, "rationale": "common cause"})
		differentials.append({"name": "GERD", "likelihood_percent": 15, "rationale": "reflux may mimic"})
	red_flags = guides.get("chest_pain", {}).get("red_flags", [])
	suggested_workup = guides.get("chest_pain", {}).get("suggested_initial_workup", [])
	dx = {
		"differentials": differentials,
		"red_flags": red_flags,
		"suggested_workup": suggested_workup,
		"notes": "Offline conservative estimate",
	}
	plan = [
		{"name": "sublingual nitroglycerin", "dose": "0.4 mg", "frequency": "q5min PRN chest pain", "notes": "if not hypotensive"},
		{"name": "aspirin", "dose": "81 mg", "frequency": "daily", "notes": "if no allergy"},
		{"name": "lifestyle: rest and avoid exertion", "dose": None, "frequency": None, "notes": "until evaluation"}
	]
	follow_up = ["ECG", "troponin", "urgent evaluation if pain persists or worsens"]
	tx = {"plan": plan, "follow_up": follow_up, "rationale": "Guideline-inspired first-line"}
	return {
		"collected": collected,
		"diagnosis": dx,
		"treatment": apply_treatment_safety(tx, collected),
	}


def run_healthcare_crew(patient_context: Dict[str, Any]) -> Dict[str, Any]:
	use_offline = os.getenv("OFFLINE_MODE", "false").lower() == "true" or not export_available
	if use_offline:
		return _offline_flow(patient_context)

	# Import task creators only when needed to avoid crewai imports during offline
	from .tasks import (
		create_symptom_collection_task,
		create_diagnosis_task,
		create_treatment_task,
	)

	collector = build_symptom_collector_agent()
	diagnoser = build_diagnosis_agent()
	treatment = build_treatment_planner_agent()

	collect_task = create_symptom_collection_task(collector, patient_context)
	crew_collect = Crew(agents=[collector], tasks=[collect_task], process=Process.sequential)
	collected = crew_collect.kickoff()
	if isinstance(collected, str):
		# output_json=True should parse, but defensively handle
		try:
			import json
			collected = json.loads(collected)
		except Exception:
			collected = {"raw": collected}

	diagnosis_task = create_diagnosis_task(diagnoser, collected)
	crew_dx = Crew(agents=[diagnoser], tasks=[diagnosis_task], process=Process.sequential)
	dx = crew_dx.kickoff()
	if isinstance(dx, str):
		try:
			import json
			dx = json.loads(dx)
		except Exception:
			dx = {"raw": dx}

	treatment_task = create_treatment_task(treatment, collected, dx)
	crew_tx = Crew(agents=[treatment], tasks=[treatment_task], process=Process.sequential)
	tx = crew_tx.kickoff()
	if isinstance(tx, str):
		try:
			import json
			tx = json.loads(tx)
		except Exception:
			tx = {"raw": tx}

	tx_safe = apply_treatment_safety(tx, collected)

	return {
		"collected": collected,
		"diagnosis": dx,
		"treatment": tx_safe,
	}