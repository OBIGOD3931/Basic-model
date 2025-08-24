# Healthcare Diagnostic Support System (CrewAI)

## Overview
A multi-agent CrewAI demo for healthcare diagnostic support:
- Symptom Collector Agent: gathers patient symptoms, history, and wearable data.
- Diagnosis Agent: synthesizes a prioritized differential diagnosis.
- Treatment Planner Agent: proposes an initial treatment plan and follow-up.

This project includes a mock LLM to run fully offline without API keys. If you later add a real LLM, you can wire it in `agents.py`.

## Requirements
- Python 3.10+

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run CLI
```bash
PYTHONPATH=src python -m healthcare_crew.cli \
  --patient "chest pain, shortness of breath" \
  --history "hypertension" \
  --wearable-json '{}'
```

## Tests
```bash
PYTHONPATH=src pytest -q
```

## Notes
- The mock LLM reads embedded JSON between `<MOCK_JSON_START>` and `<MOCK_JSON_END>` that the tasks include, ensuring deterministic structured outputs for tests and offline runs.
- Pydantic models in `src/healthcare_crew/models.py` define the structured outputs.