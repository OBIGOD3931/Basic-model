from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class WearableData(BaseModel):
	heart_rate_bpm: Optional[int] = Field(default=None)
	systolic_bp: Optional[int] = Field(default=None)
	diastolic_bp: Optional[int] = Field(default=None)
	spo2_percent: Optional[int] = Field(default=None)
	temperature_c: Optional[float] = Field(default=None)
	steps: Optional[int] = Field(default=None)
	sleep_quality: Optional[str] = Field(default=None)


class PatientSymptomReport(BaseModel):
	patient_summary: str = Field(description="Brief summary of patient narrative and key details")
	symptoms: List[str] = Field(default_factory=list)
	medical_history: List[str] = Field(default_factory=list)
	wearable_data: WearableData = Field(default_factory=WearableData)


class ConditionHypothesis(BaseModel):
	name: str
	likelihood: float = Field(ge=0.0, le=1.0)
	rationale: str


class DiagnosisSuggestion(BaseModel):
	most_likely: ConditionHypothesis
	ddx: List[ConditionHypothesis] = Field(default_factory=list)
	recommended_tests: List[str] = Field(default_factory=list)
	red_flags: List[str] = Field(default_factory=list)


class MedicationRecommendation(BaseModel):
	name: str
	dose: str
	instructions: str
	cautions: List[str] = Field(default_factory=list)


class TreatmentPlan(BaseModel):
	recommendations_summary: str
	medications: List[MedicationRecommendation] = Field(default_factory=list)
	lifestyle: List[str] = Field(default_factory=list)
	follow_up: str
	warnings: List[str] = Field(default_factory=list)