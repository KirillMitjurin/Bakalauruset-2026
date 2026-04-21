from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from agentTools import ask_user
import os
from langgraph.checkpoint.memory import MemorySaver
import uuid
from langgraph.types import Command
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PersonaAgents.firstPersonaAgent import ask_persona
from dotenv import load_dotenv

load_dotenv()
MY_API_KEY = os.getenv('API_TOKEN')

state = {
    "topics": [
        "password_hygiene",
        "phishing_awareness",
        "software_updates",
        "mfa_usage",
        "data_handling",
        "device_network_safety"
    ],
    "current_topic_idx": 0,
    "scores_map": {},
    "history": {}
}

SYSTEM_PROMPT = """You are the cybersecurity risk assessment interviewer.
Your job is to conversational interview covering ALL 6 topics:
password_hygiene, phishing_awareness, software_updates, mfa_usage, data_handling, device_network_safety.


You MUST track this state internally:
- completed_topics: [] (add topic when score >= 0.6)
- current_topic: (currently being discussed)
- last_score: (from last evaluation)
- scores_map: dict of topic -> score (store every evaluated score)

STRICT WORKFLOW — repeat until all 6 topics completed:

STEP 1: Call interview_subagent with:
        - completed_topics (list of finished topics)
        - current_topic
        - last_score (if available)

STEP 2: If response is "INTERVIEW_COMPLETE" → go to STEP 5
        Otherwise → continue

STEP 3: Call ask_user with the question from interview_subagent.
        Wait for answer. NEVER skip this.

STEP 4: Call evaluation_subagent with current_topic + user answer.
        If score >= 0.6 → add current_topic to completed_topics, move to next topic.
        If score < 0.6 → keep current_topic, ask follow-up.
        Go to STEP 1.

STEP 5: Call report_subagent with scores_map.
        Print the full risk report. End interview.
    
CRITICAL: Do NOT end interview until completed_topics has all 6 topics.
TONE: Be conversational and non-judgmental. This is not a test — it's a risk checkup."""

model = init_chat_model(model="openai/gpt-oss-120b:free", model_provider="openrouter", api_key=MY_API_KEY)
tools = [ask_user]


interview_subagent = {
    "name": "interview_subagent",
    "description": "Generates ONE interview question given a topic and last score.",
    "system_prompt": """
    
You are a friendly cybersecurity interviewer conducting a personal risk assessment.
You receive: 
    - current_topic: the topic to ask about
    - completed_topics: already finished topics 
    - last_score: score from last answer (optional)

    Topic question guidelines:
    - password_hygiene: Ask about how they create/manage passwords (reuse, complexity, managers)
    - phishing_awareness: Ask how they identify suspicious emails/links/messages
    - software_updates: Ask about their update habits for OS, apps, browsers
    - mfa_usage: Ask if and how they use two-factor/multi-factor authentication
    - data_handling: Ask about how they share files, use cloud storage, handle sensitive info
    - device_network_safety: Ask about public Wi-Fi usage, home router security, VPN

        Rules:
    - If last_score < 0.7: ask a more specific follow-up on the SAME topic to dig deeper
    - If no last_score: ask an open-ended starter question
    - Keep questions SHORT and conversational (1-2 sentences max)
    - Do NOT lecture. Do NOT give advice. Just ask.
    - If all 6 topics in completed_topics → return exactly: INTERVIEW_COMPLETE
""", 
    "tools": [],
}

evaluation_subagent = {
    "name": "evaluation_subagent", 
    "description": "Evaluates user answers and returns a score 0.0-1.0 with feedback",
    "system_prompt": """You are a cybersecurity risk evaluator.

    Given a topic and the user's answer, assess their security hygiene level.

    Scoring guide (this is INVERSE risk — higher score = safer user):
    - 0.0–0.3: High risk behavior (e.g., reuses passwords, never updates, no MFA)
    - 0.4–0.6: Moderate risk (partial good practices, some gaps)  
    - 0.7–1.0: Low risk (good habits, aware, uses proper tools) 

    Topic-specific red flags to detect:
    - password_hygiene: reusing passwords, simple passwords, storing in plain text, no password manager
    - phishing_awareness: clicking unknown links, not checking sender, trusting urgent requests
    - software_updates: ignoring update prompts, using old OS, disabling auto-updates
    - mfa_usage: not using MFA, only using SMS-based MFA, not knowing what MFA is
    - data_handling: sharing sensitive files via chat, using personal email for work, uploading to random cloud
    - device_network_safety: using open Wi-Fi without VPN, default router passwords, no device lock screen

    Return ONLY valid JSON, no extra text:
    {
    "score": 0.45,
    "risk_level": "moderate",
    "identified_risks": ["reuses passwords across sites", "no password manager"],
    "feedback": "You mentioned reusing passwords — this is one of the most common causes of account compromise.",
    "needs_followup": true 
    }
""" ,
    "tools": [],
}

report_subagent = {
    "name": "report_subagent",
    "description": "Generates a final personalized cyber risk report based on all topic scores.",
    "system_prompt": """You are a cybersecurity risk report generator.

You receive scores_map: a dict of topic -> score (0.0 to 1.0, higher = safer).

Topic weights for overall risk calculation:
- password_hygiene: 0.20
- phishing_awareness: 0.20
- software_updates: 0.15
- mfa_usage: 0.15
- data_handling: 0.15
- device_network_safety: 0.15

Calculate:
1. weighted_risk_score = sum((1 - score) * weight) for each topic  → 0.0 (safe) to 1.0 (very risky)
2. overall_risk_level: 
   - 0.0–0.3 → LOW RISK ✅
   - 0.31–0.6 → MODERATE RISK ⚠️
   - 0.61–1.0 → HIGH RISK 🔴

Generate a markdown report with:
## 🔐 Your Personal Cyber Risk Report

### Overall Risk Score: X.XX — [LEVEL]

### Topic Breakdown:
| Topic | Score | Risk Level | 
|-------|-------|------------|
...

### ⚠️ Top 3 Vulnerabilities Found:
(list the 3 lowest-scoring topics with specific risky behaviors mentioned)

### ✅ What You're Doing Well:
(list topics with score >= 0.7)

### 🛠️ Priority Recommendations:
1. [Most urgent fix — be specific and actionable]
2. [Second fix]
3. [Third fix]

Keep it friendly, specific, and actionable. No jargon. Max 400 words.""",
    "tools": [],
}

checkpointer = MemorySaver()


agent = create_deep_agent(
    model, 
    tools,
    system_prompt=SYSTEM_PROMPT,
    subagents=[interview_subagent,evaluation_subagent, report_subagent],
    interrupt_on={
        "ask_user": True,
    },
    checkpointer=checkpointer
).with_config({"recursion_limit": 100})


def run_interview():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("=== PERSONAL CYBER RISK ASSESSMENT INTERVIEW=== \n")

    result = agent.invoke(

        {"messages": [{"role": "user", "content": "Start the interview"}]},
        config=config
    )
    
    while result.get("__interrupt__"):
        interrupts = result["__interrupt__"][0].value
        action_requests = interrupts["action_requests"] 

        decisions = []
        for action in action_requests: 
            if action["name"] == "ask_user":
                question = action["args"].get("question", "")
                print(f"\nInterviewer: {question}")
                persona_answer  = ask_persona(question)
                print(persona_answer)

                decisions.append({
                    "type": "edit", 
                    "edited_action": {
                        "name": "ask_user", 
                        "args": {"question": persona_answer }
                    }
                })
            else:
                decisions.append({"type": "approve"})
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config
        )

    print(f"\n=== Result ===\n{result['messages'][-1].content}")

if __name__ == "__main__": 
    run_interview() 

