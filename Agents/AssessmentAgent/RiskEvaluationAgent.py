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
from typing import Any, Dict, List

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

def get_eits_context(scenario: str) -> str:
    docs = retriever.invoke(scenario)
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )


# Treatment decision rules is suggetion, add to the thesis.
RISK_EVALUATION_PROMPT = """
You are a cybersecurity risk evaluation agent.

Your task is to evaluate already identified and analysed cybersecurity risks.
Do NOT identify new risks.
Do NOT remove risks.
Do NOT rename risk_id.
Do NOT change the original evidence.
Do NOT change likelihood, impact, or severity.

You are given:
1. A JSON object containing analysed risks with likelihood, impact, severity, and analysis justification.
2. Relevant E-ITS context.

Use the E-ITS context only to support the evaluation and treatment recommendation.
Do NOT create new risks only because they appear in the E-ITS context.
Do NOT assume facts that are not present in the risk evidence.

Default risk appetite for this prototype:
- High severity risks are not acceptable and should normally be reduced or avoided.
- Medium severity risks should normally be reduced or retained with monitoring.
- Low severity risks may normally be retained.
- Transferred is used only when the risk can reasonably be shared with another party, such as insurance, partners, or external service providers.

For each risk, assign:
- acceptable: true or false
- priority: Low, Medium, or High
- treatment_decision: avoided, reduced, transferred, or retained
- evaluation_justification: short explanation based on severity, evidence, and, where relevant, E-ITS context

Decision rules:
- If severity is High, acceptable must be false.
- If severity is High, priority should be High.
- If severity is Medium, priority should be Medium unless the evidence suggests serious business impact.
- If severity is Low, priority should be Low.
- Most cybersecurity control weaknesses should be reduced unless there is a clear reason to avoid, transfer, or retain them.

Return ONLY valid JSON.
Do not use markdown.
Do not include explanations outside JSON.

Output format:
{
  "risks": [
    {
"risk_name": "Phishing attack",
    "category": "personnel",
    "evidence": "I always click on links in emails without verifying the sender.",
    "description": "Clicking on links in emails without verifying the sender can lead to phishing attacks.",
    "possible_consequence": "Unauthorized access to sensitive information, financial loss, and identity theft."
    "likelihood": "High",
    "impact": "High",
    "risk_level": "High",
    "analysis_justification": "The evidence indicates a high likelihood of falling for phishing attack, and the possible consequences are severe, leading to a high risk level.",
    "eits_support": "The E-ITS context supports the analysis by highlighting the prevalence of phishing attacks and the importance of verifying email senders to prevent such attacks."
    "acceptable": false,
    "priority": "High",
    "treatment_decision": "reduced",
    "evaluation_justification": "The risk is not acceptable because its risk level is high. It should be reduced by implementing security awareness training and email filtering solutions to mitigate the risk of phishing attacks."
    }
  ]
}
"""

def get_eits_context_for_risks(risks: List[Dict[str, Any]]) -> str:
    if not risks:
        return "No risks provided. No E-ITS context retrieved."

    query = "\n".join(
        f"{r.get('risk_name', '')} "
        f"{r.get('description', '')} "
        f"{r.get('possible_consequence', '')} "
        f"{r.get('category', '')} "
        f"{r.get('severity', '')} "
        f"{r.get('impact', '')} "
        f"{r.get('analysis_justification', '')}"
        for r in risks
    )

    docs = retriever.invoke(query)

    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )


model  = init_chat_model(
    model="anthropic/claude-sonnet-4.5",
    model_provider="openrouter",
    api_key=MY_API_KEY,
    temperature=0
)

def risk_evaluation_agent(analyzed_risks: Dict[str, Any]) -> Dict[str, Any]:
    risks = analyzed_risks.get("risks", [])
    analysis_input = {
    "risks": risks,
    "eits_context": get_eits_context_for_risks(risks)
}
    
    risk_analysis_response = model.invoke([
    SystemMessage(content=RISK_EVALUATION_PROMPT),
    HumanMessage(content=json.dumps(analysis_input, ensure_ascii=False))
    ])

    parsed = parse_llm_json(risk_analysis_response.content)

    if "risks" not in parsed:
        return {
            "risks": [],
            "parse_error": "Risk evaluation response did not contain 'risks'",
            "raw_response": risk_analysis_response.content
        }

    return parsed