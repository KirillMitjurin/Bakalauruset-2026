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

def get_eits_context(scenario: str) -> str:
    docs = retriever.invoke(scenario)
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )


RISK_ANALYSIS_PROMPT = """
You are a cybersecurity risk analysis agent.

Your task is to analyse already identified cybersecurity risks.
Do NOT identify new risks.
Do NOT remove risks.
Do NOT rename risk_id.
Do NOT change the original evidence.

You are given:
1. A JSON object containing identified risks.
2. Relevant E-ITS context.

Use the E-ITS context only to support classification, justification, and recommendations.
Do NOT create new risks only because they appear in the E-ITS context.
Do NOT assume facts that are not present in the risk evidence.
For each risk, assign:
- likelihood: Low, Medium, High, or Very High   
- impact: Low, Medium, High, or Very High
- analysis_justification: short explanation based on the evidence and, where relevant, E-ITS context
- eits_support: short note explaining how E-ITS context supports the analysis, or "No direct E-ITS support found"
- risk_level: Low, Medium, or High, based on the likelihood and impact using the following rules:

Severity rules:
- impact low + likehood low = risk_level low
- impact low + likehood medium = risk_level low
- impact medium + likehood low = risk_level low
- impact very high + likehood medium = risk_level high
- impact high + likehood high = risk_level high
- impact medium + likehood very high = risk_level high
- impact very high + likehood high = risk_level very high
- impact very high + likehood very high = risk_level very high
- impact high + likehood very high = risk_level very high
- every other combination = risk_level medium


Return ONLY valid JSON.

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
    }
  ]
}
"""

def get_eits_context_for_risks(risks):
    if isinstance(risks, dict):
        risks = risks.get("risks", [])

    if not risks:
        return "No risks provided. No E-ITS context retrieved."

    query = "\n".join(
        f"{r.get('risk_name', '')} {r.get('description', '')} {r.get('possible_consequence', '')} {r.get('category', '')}"
        for r in risks
        if isinstance(r, dict)
    )

    if not query.strip():
        return "No valid risk data provided. No E-ITS context retrieved."

    docs = retriever.invoke(query)
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )


model  = init_chat_model(
    model="openai/gpt-5-mini",
    model_provider="openrouter",
    api_key=MY_API_KEY,
    temperature=0
)

def risk_analysis_agent(identified_risks: Dict[str, Any]) -> Dict[str, Any]:
    risks = identified_risks.get("risks", [])

    analysis_input = {
    "risks": risks, 
    "eits_context": get_eits_context_for_risks(risks)
}
    
    risk_analysis_response = model.invoke([
    SystemMessage(content=RISK_ANALYSIS_PROMPT),
    HumanMessage(content=json.dumps(analysis_input, ensure_ascii=False))
    ])

    parsed = parse_llm_json(risk_analysis_response.content)

    if "risks" not in parsed:
        return {
            "risks": [],
            "parse_error": "Risk analysis response did not contain 'risks'",
            "raw_response": risk_analysis_response.content
        }

    return parsed