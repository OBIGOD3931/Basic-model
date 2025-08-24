from __future__ import annotations

from typing import Any

try:
	from crewai import Agent
except Exception:  # pragma: no cover - allow offline/dry-run without crewai installed
	Agent = Any  # type: ignore

from .llm_config import get_llm


def build_symptom_collector_agent() -> Agent:
	return Agent(
		role="Symptom Collector",
		goal=(
			"Interact with the patient context to extract structured symptoms, onset, duration, severity, "
			"medical history, current meds, allergies, and incorporate wearable summaries."
		),
		backstory=(
			"You are a careful clinical intake specialist. You create concise, structured summaries as JSON. "
			"Be conservative and avoid inferring facts not provided."
		),
		allow_delegation=False,
		llm=get_llm(),
	)


def build_diagnosis_agent() -> Agent:
	return Agent(
		role="Diagnosis Agent",
		goal=(
			"Use medical reasoning with clinical guidelines to produce a differential diagnosis ranked by likelihood, "
			"note red flags, and propose initial workup."
		),
		backstory=(
			"You are a clinician with strong knowledge of common conditions and triage. You output an evidence-backed "
			"differential and tests to consider, with brief rationales."
		),
		allow_delegation=False,
		llm=get_llm(),
	)


def build_treatment_planner_agent() -> Agent:
	return Agent(
		role="Treatment Planner",
		goal=(
			"Recommend initial management options (medications, lifestyle, follow-up), check for allergies and duplicates, "
			"and flag any safety issues."
		),
		backstory=(
			"You are a cautious pharmacist/clinician hybrid focusing on first-line options and safety. "
			"Prefer non-pharmacologic options when appropriate."
		),
		allow_delegation=False,
		llm=get_llm(),
	)