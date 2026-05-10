import json
import langchain
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

from AssessmentAgent.utils import parse_llm_json

load_dotenv()
MY_API_KEY = os.getenv('API_TOKEN')

def validate_agents_data_transffering(data):
    identified_risks = data["identified_risks"]["risks"]
    analysed_risks = data["analysed_risks"]["risks"]
    evaluated_risks = data["evaluated_risks"]["risks"]


    shared_fields = ["risk_name", "category", "evidence", "description", "possible_consequence"]
    analysed_and_evaluated_shared_fields = ["risk_name", "category", "description", "possible_consequence", "likelihood","impact", "analysis_justification", "eits_support"]
    errors = []

    for ident_riks in identified_risks:
        for anal_risk in analysed_risks:
            if all(ident_riks.get(field) == anal_risk.get(field) for field in shared_fields):
                break
        else:
            errors.append(f"Risk '{ident_riks.get('risk_name', 'unknown')}' not found in analysed risks.")
    
    for analysed_riks in analysed_risks:
        for eval_risk in evaluated_risks:
            if all(analysed_riks.get(field) == eval_risk.get(field) for field in analysed_and_evaluated_shared_fields):
                break
        else:
            errors.append(f"Risk '{analysed_riks.get('risk_name', 'unknown')}' not found in evaluated risks.")

    return errors

#Evaluation results from test runs of the agents, used for validating the data transfer between agents.    
for i in range(1,6):
    with open(f"evaluation_claude_results_test{i}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for scenario in data:
        print(f"Scenario {scenario['scenario_id']}:")
        validation_errors = validate_agents_data_transffering(scenario)
        if validation_errors:
            print("Data transfer validation errors:")
            for error in validation_errors:
                print(f"- {error}")
        else:
            print("Data transfer validation passed without errors.")

CONSISTENCY_CHECK_PROMPT = """
You are a semantic risk matching assistant.

You are given two lists of cybersecurity risks produced for two scenarios.
Scenario 2 is based on Scenario 1 and adds more context, so risks from Scenario 1 should normally still appear in Scenario 2.

Your task:
Find which risks from Scenario 1 are NOT present in Scenario 2.

Important rules:
- Compare risks semantically, not only by exact name.
- If two risks have different names but describe the same underlying security problem, treat them as the same risk.
- Use risk_id if available, but do not rely only on risk_id.
- Compare risk_name, evidence, description, possible_consequence
- New risks in Scenario 2 are expected and must be ignored.
- Do NOT judge whether Scenario 2 contains good new risks.
- Do NOT create new risks.
- Do NOT change risk names.
- Only report risks from Scenario 1 that are missing in Scenario 2.

Scenario 1 risks:
SCENARIO_1_RISKS_PLACEHOLDER

Scenario 2 risks:
SCENARIO_2_RISKS_PLACEHOLDER

Return ONLY valid JSON.
Do not use markdown.
Do not include explanations outside JSON.

Output format:
{
  "missing_risks": [
    {
      "scenario_1_risk_id": "...",
      "scenario_1_risk_name": "...",
      "reason": "Short explanation why no semantically matching risk was found in Scenario 2"
    }
  ],
  "matched_risks": [
    {
      "scenario_1_risk_id": "...",
      "scenario_1_risk_name": "...",
      "scenario_2_risk_id": "...",
      "scenario_2_risk_name": "...",
      "reason": "Short explanation why these risks are semantically the same"
    }
  ]
}
"""

model = init_chat_model(model="google/gemma-4-31b-it", model_provider="openrouter", api_key=MY_API_KEY, temperature=0)

def evaluate_consistency(index, scenario_number):    
    with open(f"evaluation_claude_results_test{index}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    scenario_1_risks = data[scenario_number]["identified_risks"]["risks"]
    scenario_2_risks = data[scenario_number + 1]["identified_risks"]["risks"]

    prompt = CONSISTENCY_CHECK_PROMPT \
        .replace("SCENARIO_1_RISKS_PLACEHOLDER", json.dumps(scenario_1_risks, ensure_ascii=False)) \
        .replace("SCENARIO_2_RISKS_PLACEHOLDER", json.dumps(scenario_2_risks, ensure_ascii=False))
    
    response = model.invoke([HumanMessage(content=prompt)])


    print("Consistency Check Response:")
    print(response.content) 
    parsed_json = parse_llm_json(response.content) 
    print(f"Missing risks: {len(parsed_json['missing_risks'])}")
    print(f"Matched risks: {len(parsed_json['matched_risks'])}")
    print(f"Scenario {scenario_number + 1} risks: {len(scenario_1_risks)}")

    missing_count = len(parsed_json.get("missing_risks", []))
    matched_count = len(parsed_json.get("matched_risks", []))
    scenario_1_count = len(scenario_1_risks)

    retention_score = (scenario_1_count - missing_count) / scenario_1_count if scenario_1_count > 0 else 0


    with open(f"consistency_results_claude{index}.txt", "a", encoding="utf-8") as f:
        f.write(f"Test file: evaluation_claude_results_test{index}.json\n")
        f.write(f"Scenario {scenario_number + 1} -> Scenario {scenario_number + 2}\n")
        f.write("-" * 80 + "\n")
        f.write(response.content)
        f.write("\n\n")
        f.write(f"Missing risks: {missing_count}\n")
        f.write(f"Matched risks: {matched_count}\n")
        f.write(f"Scenario {scenario_number + 1} risks: {scenario_1_count}\n")
        f.write(f"Retention score: {retention_score:.3f}\n")
        f.write("\n\n")



for index in range(1,6):
    for scenario_number in range(0,4):
        print(f"Evaluating consistency between scenarios {scenario_number + 1} and {scenario_number + 2} for test{index}.json")
        evaluate_consistency(index, scenario_number)
