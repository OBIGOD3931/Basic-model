from __future__ import annotations

from typing import Dict, Optional

from crewai import Task

from .models import (
	DiagnosisSuggestion,
	MedicationRecommendation,
	PatientSymptomReport,
	TreatmentPlan,
	WearableData,
	ConditionHypothesis,
)
from .llm.mock_llm import MOCK_START, MOCK_END


def _embed_mock_json(json_str: str) -> str:
	return f"\nWhen running in MOCK mode, output exactly and only the JSON below.\n{MOCK_START}\n{json_str}\n{MOCK_END}\n"


# Public helpers for deterministic offline logic

def synthesize_symptom_report(
	patient_text: str,
	medical_history_text: Optional[str] = None,
	wearable: Optional[Dict] = None,
) -> PatientSymptomReport:
	wearable_data = WearableData(**(wearable or {}))
	return PatientSymptomReport(
		patient_summary=patient_text.strip(),
		symptoms=[s.strip() for s in patient_text.split(',') if s.strip()],
		medical_history=[h.strip() for h in (medical_history_text or "").split(',') if h.strip()],
		wearable_data=wearable_data,
	)


def infer_mock_diagnosis(report: PatientSymptomReport) -> DiagnosisSuggestion:
	symptoms_lower = " ".join(report.symptoms).lower()
	ddx: list[ConditionHypothesis] = []
	red_flags: list[str] = []
	recommended_tests: list[str] = []

	if "chest" in symptoms_lower and "pain" in symptoms_lower:
		ddx.append(ConditionHypothesis(name="Angina", likelihood=0.6, rationale="Exertional chest pain"))
		ddx.append(ConditionHypothesis(name="GERD", likelihood=0.2, rationale="Burning quality, post-prandial"))
		ddx.append(ConditionHypothesis(name="Musculoskeletal pain", likelihood=0.2, rationale="Reproducible tenderness"))
		red_flags.extend(["Severe chest pain", "Dyspnea", "Syncope"])
		recommended_tests.extend(["ECG", "Troponin", "CXR"])
	else:
		ddx.append(ConditionHypothesis(name="Non-specific", likelihood=0.7, rationale="Insufficient data"))
		recommended_tests.append("Basic labs")

	most_likely = max(ddx, key=lambda c: c.likelihood)
	return DiagnosisSuggestion(
		most_likely=most_likely,
		ddx=ddx,
		recommended_tests=recommended_tests,
		red_flags=red_flags,
	)


def build_mock_treatment(dx: DiagnosisSuggestion) -> TreatmentPlan:
	meds: list[MedicationRecommendation] = []
	lifestyle: list[str] = []
	warnings: list[str] = []
	follow_up = "Follow up with primary care in 1-2 weeks or earlier if worse."

	name = dx.most_likely.name.lower()
	if "angina" in name:
		meds.append(
			MedicationRecommendation(
				name="Aspirin",
				dose="81 mg daily",
				instructions="Take with food unless contraindicated",
				cautions=["Bleeding risk", "Allergy"],
			)
		)
		lifestyle.extend(["Avoid exertion until evaluated", "Seek urgent care for severe symptoms"])
		warnings.extend(["Chest pain with dyspnea/syncope requires ED evaluation"])
	elif "non-specific" in name:
		lifestyle.append("Symptom diary and hydration")
	else:
		lifestyle.append("General supportive care")

	return TreatmentPlan(
		recommendations_summary=f"Initial plan for {dx.most_likely.name}",
		medications=meds,
		lifestyle=lifestyle,
		follow_up=follow_up,
		warnings=warnings,
	)


# Task builders

def build_symptom_collection_task(
	agent,
	patient_text: str,
	medical_history_text: Optional[str] = None,
	wearable: Optional[Dict] = None,
) -> Task:
	report = synthesize_symptom_report(
		patient_text=patient_text,
		medical_history_text=medical_history_text,
		wearable=wearable,
	)
	mock_json = report.model_dump_json()

	description = (
		"Collect and structure the patient's information into a JSON report with: "
		"patient_summary, symptoms, medical_history, wearable_data.\n"
		"Extract symptom keywords; list history items from the provided history text.\n"
	)
	description += _embed_mock_json(mock_json)

	return Task(
		description=description,
		agent=agent,
		expected_output=(
			"A valid JSON object matching the PatientSymptomReport schema."
		),
		output_pydantic=PatientSymptomReport,
	)


def build_diagnosis_task(agent, report: PatientSymptomReport) -> Task:
	dx = infer_mock_diagnosis(report)
	mock_json = dx.model_dump_json()

	description = (
		"Based on the structured symptom report, produce a prioritized differential diagnosis.\n"
		"Include most_likely, ddx (list), recommended_tests, red_flags."
	)
	description += _embed_mock_json(mock_json)

	return Task(
		description=description,
		agent=agent,
		expected_output="A valid JSON object matching the DiagnosisSuggestion schema.",
		output_pydantic=DiagnosisSuggestion,
	)


def build_treatment_task(agent, dx: DiagnosisSuggestion) -> Task:
	plan = build_mock_treatment(dx)
	mock_json = plan.model_dump_json()

	description = (
		"Recommend an initial safe treatment plan including medications, lifestyle, follow_up, warnings."
	)
	description += _embed_mock_json(mock_json)

	return Task(
		description=description,
		agent=agent,
		expected_output="A valid JSON object matching the TreatmentPlan schema.",
		output_pydantic=TreatmentPlan,
	)