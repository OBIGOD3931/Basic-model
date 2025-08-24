import os
from typing import Optional

try:
	from crewai import LLM
except Exception:  # pragma: no cover - allows import before deps are installed
	LLM = object  # type: ignore


def get_llm() -> "LLM":
	"""
	Create a CrewAI LLM configured from environment variables.

	Environment:
	- LLM_MODEL: model id (default: gpt-4o-mini)
	- USE_OLLAMA: if true, force ollama/<model>
	- OLLAMA_MODEL: ollama model id (default: llama3.1)
	- LLM_TEMPERATURE: decoding temperature (default: 0.1)
	"""
	model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
	use_ollama: bool = os.getenv("USE_OLLAMA", "false").lower() == "true"
	if use_ollama and not model.startswith("ollama/"):
		model = f"ollama/{os.getenv('OLLAMA_MODEL', 'llama3.1')}"

	temp_str: str = os.getenv("LLM_TEMPERATURE", "0.1")
	try:
		temperature: float = float(temp_str)
	except ValueError:
		temperature = 0.1

	return LLM(model=model, temperature=temperature)


def offline_mode_enabled() -> bool:
	return os.getenv("OFFLINE_MODE", "false").lower() == "true"


def have_any_api_key() -> bool:
	keys = [
		"OPENAI_API_KEY",
		"ANTHROPIC_API_KEY",
		"AZURE_OPENAI_API_KEY",
		"GROQ_API_KEY",
		"MISTRAL_API_KEY",
	]
	return any(os.getenv(k) for k in keys)