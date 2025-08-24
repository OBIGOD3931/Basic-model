import json
import subprocess
import sys
import os


def test_cli_smoke():
	cmd = [sys.executable, "-m", "healthcare_crew.cli", "--patient", "chest pain, shortness of breath", "--history", "hypertension", "--wearable-json", "{}"]
	env = os.environ.copy()
	env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
	result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
	data = json.loads(result.stdout)
	assert "symptom_report" in data
	assert "diagnosis" in data
	assert "treatment_plan" in data