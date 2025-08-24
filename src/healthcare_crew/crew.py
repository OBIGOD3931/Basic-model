from __future__ import annotations

from typing import Dict, Optional

from crewai import Crew, Process

from .agents import AgentsFactory
from .tasks import (
	build_diagnosis_task,
	build_symptom_collection_task,
	build_treatment_task,
	synthesize_symptom_report,
	infer_mock_diagnosis,
	build_mock_treatment,
)
from .models import PatientSymptomReport, DiagnosisSuggestion, TreatmentPlan


def build_healthcare_crew(
	patient_text: str,
	medical_history_text: Optional[str] = None,
	wearable: Optional[Dict] = None,
):
	factory = AgentsFactory()

	symptom_agent = factory.build_symptom_collector()
	dx_agent = factory.build_diagnosis_agent()
	plan_agent = factory.build_treatment_planner()

	# Precompute deterministic objects to embed into tasks (for mock runs)
	report_obj: PatientSymptomReport = synthesize_symptom_report(
		patient_text=patient_text,
		medical_history_text=medical_history_text,
		wearable=wearable,
	)
	dx_obj: DiagnosisSuggestion = infer_mock_diagnosis(report_obj)
	plan_obj: TreatmentPlan = build_mock_treatment(dx_obj)

	task1 = build_symptom_collection_task(
		agent=symptom_agent,
		patient_text=patient_text,
		medical_history_text=medical_history_text,
		wearable=wearable,
	)
	task2 = build_diagnosis_task(agent=dx_agent, report=report_obj)
	task3 = build_treatment_task(agent=plan_agent, dx=dx_obj)

	crew = Crew(
		agents=[symptom_agent, dx_agent, plan_agent],
		tasks=[task1, task2, task3],
		process=Process.sequential,
		verbose=False,
	)
	return crew


def run_pipeline(
	patient_text: str,
	medical_history_text: Optional[str] = None,
	wearable: Optional[Dict] = None,
) -> Dict[str, object]:
	# Precompute deterministic fallback outputs
	report_fallback: PatientSymptomReport = synthesize_symptom_report(
		patient_text=patient_text,
		medical_history_text=medical_history_text,
		wearable=wearable,
	)
	dx_fallback: DiagnosisSuggestion = infer_mock_diagnosis(report_fallback)
	plan_fallback: TreatmentPlan = build_mock_treatment(dx_fallback)
	fallback_outputs: Dict[str, object] = {
		"symptom_report": report_fallback.model_dump(),
		"diagnosis": dx_fallback.model_dump(),
		"treatment_plan": plan_fallback.model_dump(),
	}

	try:
		crew = build_healthcare_crew(patient_text, medical_history_text, wearable)
		_ = crew.kickoff()
		# Normalize outputs
		report: PatientSymptomReport = crew.tasks[0].output
		dx: DiagnosisSuggestion = crew.tasks[1].output
		plan: TreatmentPlan = crew.tasks[2].output
		return {
			"symptom_report": report.model_dump(),
			"diagnosis": dx.model_dump(),
			"treatment_plan": plan.model_dump(),
		}
	except Exception:
		# Fall back to deterministic outputs to ensure offline success
		return fallback_outputs