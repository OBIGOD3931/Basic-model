# Healthcare Diagnostic Support System (CrewAI)

A multi-agent prototype built with CrewAI to assist clinicians by collecting symptoms, suggesting differential diagnoses, and proposing treatment plans with basic safety checks.

## Quickstart

1. Create and activate a virtualenv
```bash
python3 -m venv .venv && source .venv/bin/activate
```

2. Install dependencies
- Online/full (CrewAI + LLM providers):
```bash
pip install -r requirements.txt
```
- Offline demo (no CrewAI/LLM calls required):
```bash
pip install -r requirements.offline.txt
```

3. Configure LLM access (choose one)
- OpenAI: set `OPENAI_API_KEY` and optionally `LLM_MODEL` (default `gpt-4o-mini`)
- Ollama: set `USE_OLLAMA=true` and optionally `OLLAMA_MODEL` (default `llama3.1`)
- Offline dry run: pass `--dry-run` to the CLI to avoid LLM calls

You can copy `.env.example` to `.env` and fill in your values.

4. Run the demo
```bash
PYTHONPATH=src python3 -m healthcare_ai.cli demo --dry-run
```

To use a real model (requires keys):
```bash
PYTHONPATH=src python3 -m healthcare_ai.cli demo
```

## CLI

- Run with your own patient JSON and optional wearable JSON
```bash
PYTHONPATH=src python3 -m healthcare_ai.cli run --patient-file data/sample_patient.json --wearable-file data/sample_wearable.json --dry-run
```

## Notes
- This is a research prototype and not a medical device. It does not provide medical advice. Always consult licensed professionals.
- Safety checks are simplistic and illustrative only.