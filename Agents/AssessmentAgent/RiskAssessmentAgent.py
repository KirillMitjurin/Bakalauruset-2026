from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import os
from langgraph.checkpoint.memory import MemorySaver
import uuid
from langgraph.types import Command
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    "I use the same password for every service. I also use this password for my work accounts. At work, I use a company-issued computer where I have administrative privileges.", 
    "I use the same password for every service. I use a company issued computer where I have administrative privileges. I also use a company provided Android phone for work-related tasks, such as accessing emails and documents."
    "I use the same password for every service. I use a company issued computer where I have administrative privileges. I also use a company provided Android phone for work tasks. Our company relies on cloud services such as Google Drive to store and share data.",
    "I use the same password for every service. I use a company issued computer where I have administrative privileges. I also use a company provided Android phone for work tasks. Our company relies on cloud services such as Google Drive to store and share data. Employees regularly access these systems to collaborate and manage files."
]


def get_eits_context(scenario: str) -> str:
    docs = retriever.invoke(scenario)
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )

FIXED_EITS_QUERIES = [
    "password security policy",
    "administrative privileges",
    "mobile device security",
    "cloud storage security",
    "employee responsibilities"
]

def get_fixed_eits_context() -> str:
    all_docs = []
    seen = set()
    for query in FIXED_EITS_QUERIES:
        docs = retriever.invoke(query)
        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in all_docs
    )


EITS_CONTEXT = get_fixed_eits_context()

RISK_IDENTIFIER_PROMPT = """You are a cybersecurity risk identifier.

CRITICAL RULES:
- Only identify risks DIRECTLY supported by the facts.
- Do NOT assume missing information.
- Do NOT infer additional risks based on general knowledge.
- Do NOT generate risks unless explicitly supported by a fact.
- Each risk MUST correspond to exactly ONE fact.
- If no fact supports the risk → DO NOT include it.
- Do NOT generate compliance-only risks.
- Do NOT generate training, policy, or MFA risks unless explicitly stated.

Facts:
FACTS_PLACEHOLDER

E-ITS context (for explanation only, NOT for generating risks):
EITS_PLACEHOLDER

Task:
For each fact, generate exactly ONE risk.

Output:
{
  "risks": [
    {
      "risk_name": "...",
      "category": "...",
      "evidence": "...",
      "description": "...",
      "possible_consequence": "...",
      "eits_reference": []
    }
  ]
}
}"""

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

# RISK_IDENTIFIER_PROMPT = """
# You are given a list of security-relevant facts.
# For each fact (and combination of facts), identify ALL applicable risks.
# Do not skip risks just because other facts are present.

# Important:
# - Treat each fact independently
# - Also consider interactions between facts (e.g. password reuse + admin privileges = extra risk)
# - Do not let new facts overshadow existing ones


# All risks MUST be assigned to exactly one of the following categories:

# - organisational_structure
# - processes_and_procedures
# - administrative_routines
# - personnel
# - physical_environment
# - system_configuration
# - hardware_software_communication
# - external_dependencies

# Your output should be a JSON array of identified risks, where each risk is represented as an object with the following structure:
# {
#   "risks": [
#     {
#       "risk_name": "Password reuse",
#       "category": "authentication",
#       "evidence": "I use the same password for every service",
#       "description": "The same password is used across multiple services.",
#       "possible_consequence": "Credential stuffing or account takeover."
#     }
#   ]
# }

# Facts: {facts}
# """

RISK_IDENTIFIER_PROMPT = """You are a cybersecurity risk identifier.

CRITICAL: Respond ONLY with a valid JSON object. No markdown, no explanations, no headers.
Your entire response must start with { and end with }.

Facts to analyze:
FACTS_PLACEHOLDER

E-ITS context:
EITS_PLACEHOLDER

For each fact (and combination of facts), identify ALL applicable risks.
Do not skip risks just because other facts are present.

All risks MUST be assigned to exactly one of the following categories:
- organisational_structure
- processes_and_procedures
- administrative_routines
- personnel
- physical_environment
- system_configuration
- hardware_software_communication
- external_dependencies

Required output format:
{
  "risks": [
    {
      "risk_name": "Password reuse",
      "category": "processes_and_procedures",
      "evidence": "user reuses same password across all services",
      "description": "The same password is used across multiple services.",
      "possible_consequence": "Credential stuffing or account takeover."
    }
  ]
}
"""

FACT_NORMALIZER_PROMPT = """You are given a list of security facts extracted from a user description.
Your job is to normalize and deduplicate them into canonical forms.

Rules:
- Merge facts that mean the same thing into one canonical fact
- Use consistent phrasing: "User X" or "Company Y"
- Keep facts atomic
- Do not add facts that were not in the input

Example:
Input: ["uses same password everywhere", "password reused across services", "one password for all accounts"]
Output: ["user reuses the same password across all services"]

Input facts:
FACTS_PLACEHOLDER

Output ONLY this JSON:
{
  "facts": ["normalized fact 1", "normalized fact 2"]
}"""

def parse_llm_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {"risks": [], "parse_error": "empty output"}

    text = raw.strip()

    try:
        decoded = json.loads(text)
        if isinstance(decoded, str):
            text = decoded.strip()
        elif isinstance(decoded, dict):
            return decoded
    except json.JSONDecodeError:
        pass

    text = text.replace("```json", "").replace("```", "").strip()

    text = text.replace("\\n", "\n").replace('\\"', '"').replace("\\t", "\t")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_part = match.group(0)
        try:
            return json.loads(json_part)
        except json.JSONDecodeError as e:
            return {
                "risks": [],
                "parse_error": str(e),
                "raw": raw
            }

    return {
        "risks": [],
        "parse_error": "no JSON object found",
        "raw": raw
    }

model = init_chat_model(model="deepseek/deepseek-chat", model_provider="openrouter", api_key=MY_API_KEY, temperature=0)


models_outputs = []
EITS_CONTEXT = get_fixed_eits_context()

for scenario in scenarios:
    facts_response = model.invoke([
        SystemMessage(content=RISK_EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=scenario)
    ])
    facts = parse_llm_json(facts_response.content).get("facts", [])

    facts_normalization_response = model.invoke([
        SystemMessage(content=FACT_NORMALIZER_PROMPT),
        HumanMessage(content=FACT_NORMALIZER_PROMPT.replace("FACTS_PLACEHOLDER", json.dumps(facts, ensure_ascii=False)))
    ])
    normalized_facts = parse_llm_json(facts_normalization_response.content).get("facts", [])

    prompt = RISK_IDENTIFIER_PROMPT \
        .replace("FACTS_PLACEHOLDER", json.dumps(normalized_facts, ensure_ascii=False)) \
        .replace("EITS_PLACEHOLDER", EITS_CONTEXT)

    risk_response = model.invoke([
        HumanMessage(content=prompt)
    ])

    parsed_output = parse_llm_json(risk_response.content)
    
    if "parse_error" in parsed_output:
        print(f"[WARNING] Parse failed. Raw output:\n{parsed_output.get('raw', '')[:300]}")
    
    models_outputs.append(parsed_output)

#     prompt = f"""
# System description:
# {scenario}

# Relevant E-ITS context:
# {eits_context}

# Identify cybersecurity risks based on the system description.
# Use the E-ITS context when relevant.
# """
#     response = model.invoke([
#         SystemMessage(content=RISK_IDENTIFIER_SYSTEM_PROMPT),
#         HumanMessage(content=prompt)
#     ])
#     output = response.content
#     parsed_output = parse_llm_json(output)
#     models_outputs.append(parsed_output)

print(json.dumps(models_outputs, indent=4))


for i in range(len(models_outputs)):
    for j in range(i+1, len(models_outputs)):
        risks_i = set(risk["risk_name"] for risk in models_outputs[i].get("risks", []))
        risks_j = set(risk["risk_name"] for risk in models_outputs[j].get("risks", []))
        overlap = risks_i.intersection(risks_j)
        print(f"Overlap between scenario {i} and {j}: {overlap}")
