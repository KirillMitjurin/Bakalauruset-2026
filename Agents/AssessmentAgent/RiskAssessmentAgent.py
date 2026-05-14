from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import os
from langgraph.checkpoint.memory import MemorySaver
import uuid
from langgraph.types import Command
import sys
from RiskEvaluationAgent import risk_evaluation_agent
from RiskAnalysisAgent import risk_analysis_agent
from utils import parse_llm_json 

from PersonaAgents.firstPersonaAgent import ask_persona
from dotenv import load_dotenv
import json 
import json.decoder

from langchain_core.messages import SystemMessage, HumanMessage
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from typing import Any, Dict


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "eits.pdf")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)


embeddings = FastEmbedEmbeddings()

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

load_dotenv()
MY_API_KEY = os.getenv('API_TOKEN')

scenarios = [
    "I use the same password for every service." , 
    "I use the same password for every service. At work, I use a company-issued computer where I have administrative privileges.", 
    "I use the same password for every service. I use a company issued computer where I have administrative privileges. I also use a company provided Android phone for work-related tasks, such as accessing emails and documents.",
    "I use the same password for every service. I use a company issued computer where I have administrative privileges. I also use a company provided Android phone for work tasks. Our company relies on cloud services such as Google Drive to store and share data.",
    "I use the same password for every service. I use a company issued computer where I have administrative privileges. I also use a company provided Android phone for work tasks. Our company relies on cloud services such as Google Drive to store and share data. Employees regularly access these systems to collaborate and manage files."
]


def get_eits_context(scenario: str, facts: list[str]) -> str:
    query = scenario + "\n" + "\n".join(facts)
    docs = retriever.invoke(query)
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )

RISK_EXTRACTOR_SYSTEM_PROMPT = """Extract all security-relevant facts from the description as a list.
Each fact should be atomic (one thing per fact).

Output JSON:
{
  "facts": [
    "user reuses same password across all services",
    "password is used for work accounts",
    "user has administrative privileges on work computer",
    "user has company Android phone for work tasks"
  ]
}

"""

RISK_IDENTIFIER_PROMPT = """
You are given a list of security-relevant facts.
For each fact (and combination of facts), identify ALL applicable risks.
Do not skip risks just because other facts are present.

FACTS: FACTS_PLACEHOLDER
E-ITS context: EITS_PLACEHOLDER
Important:
- Treat each fact independently
- Also consider interactions between facts (e.g. password reuse + admin privileges = extra risk)
- Do not let new facts overshadow existing ones

All risks MUST be assigned to exactly one of the following categories:

- organisational_structure
- processes_and_procedures
- administrative_routines
- personnel
- physical_environment
- system_configuration
- hardware_software_communication
- external_dependencies

Your output should be a JSON object of identified risks, where each risk is represented as an object with the following structure:

Return ONLY one valid JSON object.
The root must be an object with the key "risks".
Do NOT return a JSON array as the root.
Your response must start with { and end with }.
{
  "risks": [
    {
      "risk_name": "Phishing attack",
      "category": "personnel",
      "evidence": "I always click on links in emails without verifying the sender.",
      "description": "Clicking on links in emails without verifying the sender can lead to phishing attacks.",
      "possible_consequence": "Unauthorized access to sensitive information, financial loss, and identity theft."
    }
  ]
}

"""


model = init_chat_model(model="openai/gpt-5-mini", model_provider="openrouter", api_key=MY_API_KEY, temperature=0)



def evaluate_scenarios():
    all_results = []

    for i, scenario in enumerate(scenarios, start=1):
        facts_response = model.invoke([
            SystemMessage(content=RISK_EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=scenario)
        ])
        facts = parse_llm_json(facts_response.content).get("facts", [])

        eits_context = get_eits_context(scenario, facts)
        prompt = RISK_IDENTIFIER_PROMPT \
            .replace("FACTS_PLACEHOLDER", json.dumps(facts, ensure_ascii=False)) \
            .replace("EITS_PLACEHOLDER", eits_context)

        risk_response = model.invoke([
            HumanMessage(content=prompt)
        ])

        identified_risks = parse_llm_json(risk_response.content)

        if isinstance(identified_risks, list):
            identified_risks = {"risks": identified_risks}

        if "parse_error" in identified_risks:
            print(f"[WARNING] Scenario {i}: parse failed.")
            print(identified_risks.get("raw", "")[:300])

        analysed_risks = risk_analysis_agent(identified_risks)
        evaluated_risks = risk_evaluation_agent(analysed_risks)

        scenario_result = {
            "scenario_id": i,
            "scenario": scenario,
            "facts": facts,
            "identified_risks": identified_risks,
            "analysed_risks": analysed_risks,
            "evaluated_risks": evaluated_risks
        }

        all_results.append(scenario_result)
    return all_results

all_results = evaluate_scenarios()

print(json.dumps(all_results, indent=4, ensure_ascii=False))

with open(f"evaluation_gpt5_results_test{5}.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

