from __future__ import annotations

from typing import Optional

from crewai import Agent

from .llm.mock_llm import MockChatModel


def get_default_llm() -> MockChatModel:
	# For offline runs we return the mock model
	return MockChatModel()


class AgentsFactory:
	def __init__(self, llm: Optional[object] = None) -> None:
		self.llm = llm or get_default_llm()

	def build_symptom_collector(self) -> Agent:
		return Agent(
			role="Symptom Collector",
			goal=(
				"Interview patient, consolidate symptoms, history, and wearable data into a structured report."
			),
			backstory=(
				"You are a compassionate clinical intake specialist. You summarize clearly and avoid assumptions."
			),
			llm=self.llm,
			allow_delegation=False,
			verbose=False,
		)

	def build_diagnosis_agent(self) -> Agent:
		return Agent(
			role="Diagnosis Specialist",
			goal=(
				"Apply medical guidelines and heuristics to propose a prioritized differential diagnosis."
			),
			backstory=(
				"You are a careful diagnostician. You reason with safety in mind and cite red flags."
			),
			llm=self.llm,
			allow_delegation=False,
			verbose=False,
		)

	def build_treatment_planner(self) -> Agent:
		return Agent(
			role="Treatment Planner",
			goal=(
				"Recommend safe initial management, medications if appropriate, lifestyle steps, and follow-up."
			),
			backstory=(
				"You plan conservative, evidence-aligned care. You highlight when to escalate to urgent care."
			),
			llm=self.llm,
			allow_delegation=False,
			verbose=False,
		)