from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class WearableSummary:
	average_resting_hr: Optional[float]
	average_spo2: Optional[float]
	average_temp_c: Optional[float]
	steps_last_24h: Optional[int]
	flags: List[str]


def parse_wearable_json(raw: Optional[str | Dict[str, Any]]) -> WearableSummary:
	"""
	Parse wearable JSON content. Accepts a JSON string or loaded dict.
	Expected top-level keys (best-effort): heart_rate, spo2, temperature_c, steps.
	"""
	if raw is None:
		return WearableSummary(None, None, None, None, [])
	if isinstance(raw, str):
		try:
			data = json.loads(raw)
		except Exception:
			return WearableSummary(None, None, None, None, ["invalid_json"])
	else:
		data = raw

	def safe_avg(values: Optional[List[float]]) -> Optional[float]:
		if not values:
			return None
		try:
			nums = [float(v) for v in values if v is not None]
			return round(sum(nums) / len(nums), 2) if nums else None
		except Exception:
			return None

	hr = safe_avg(data.get("heart_rate"))
	spo2 = safe_avg(data.get("spo2"))
	temp = safe_avg(data.get("temperature_c"))
	steps_list = data.get("steps") or []
	steps = int(steps_list[-1]) if isinstance(steps_list, list) and steps_list else None

	flags: List[str] = []
	if hr is not None and hr > 100:
		flags.append("tachycardia")
	if spo2 is not None and spo2 < 92:
		flags.append("low_spo2")
	if temp is not None and temp >= 38.0:
		flags.append("fever")
	return WearableSummary(hr, spo2, temp, steps, flags)


# Minimal guideline knowledge base (illustrative)
GUIDELINES: Dict[str, Dict[str, Any]] = {
	"chest_pain": {
		"red_flags": [
			"sudden severe chest pain",
			"pain radiating to arm or jaw",
			"shortness of breath",
			"syncope",
		],
		"suggested_initial_workup": [
			"ECG",
			"troponin",
			"oxygen saturation",
			"blood pressure",
		],
		"differentials": [
			"acute coronary syndrome",
			"pulmonary embolism",
			"pneumothorax",
			"musculoskeletal pain",
			"GERD",
		],
	},
	"fever": {
		"red_flags": [
			"temperature >= 40C",
			"confusion",
			"neck stiffness",
			"persistent tachycardia",
		],
		"suggested_initial_workup": [
			"CBC",
			"CRP",
			"urinalysis",
			"chest X-ray if respiratory symptoms",
		],
		"differentials": [
			"viral infection",
			"bacterial infection",
			"drug fever",
			"autoimmune flare",
		],
	},
}


def lookup_guidelines(symptoms: List[str]) -> Dict[str, Any]:
	matched: Dict[str, Any] = {}
	for key, info in GUIDELINES.items():
		if any(key in s.lower() for s in symptoms):
			matched[key] = info
	return matched


def safety_check_medications(recommendations: List[Dict[str, Any]], allergies: List[str], meds: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
	"""
	Basic safety check: filter out meds containing allergens or duplicates of current meds.
	Returns (safe_recommendations, safety_flags)
	"""
	flags: List[str] = []
	safe: List[Dict[str, Any]] = []

	lower_allergies = [a.lower() for a in allergies]
	lower_current = [m.lower() for m in meds]

	for rec in recommendations:
		name = str(rec.get("name", "")).lower()
		if any(a in name for a in lower_allergies):
			flags.append(f"allergy_conflict:{rec.get('name')}")
			continue
		if any(name == m for m in lower_current):
			flags.append(f"duplicate_current_med:{rec.get('name')}")
			continue
		safe.append(rec)

	return safe, flags


def apply_treatment_safety(plan_json: Dict[str, Any], collected: Dict[str, Any]) -> Dict[str, Any]:
	recs = plan_json.get("plan") or []
	allergies = collected.get("allergies") or []
	current_meds = collected.get("current_medications") or []
	safe_recs, flags = safety_check_medications(recs, allergies, current_meds)
	return {**plan_json, "plan": safe_recs, "safety_flags": flags}