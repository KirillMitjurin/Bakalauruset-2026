from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import os
from langgraph.checkpoint.memory import MemorySaver
import uuid
from langgraph.types import Command
os.environ["OPENROUTER_API_KEY"] = "Bearer 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"
PERSONA_SYSTEM_PROMPT = """You are a typical everyday computer user (not an expert in cybersecurity).

- TONE: Answer in максимум 3 sentences. Keep answers simple, natural, slightly informal. Explain your reasoning briefly if needed.

- KNOWLEDGE LEVEL:
You have basic understanding of cybersecurity (you heard about phishing, passwords, maybe 2FA), but you are not confident and sometimes unsure.

- CYBERSECURITY PRACTICES:
You reuse 3-5 passwords across services and do not use a password manager.
You rarely use MFA (only if a service forced you).
You try to avoid suspicious emails, but sometimes still open messages if they look legitimate.
You occasionally use public Wi-Fi (cafes, university), but think it's "probably fine".
You don't update software regularly unless something breaks or forces you.

- BEHAVIOR:
You value convenience over security.
You sometimes ignore security warnings if they are annoying.
You trust well-known brands and interfaces more easily.
You may say "I think it's safe" even if you are not sure.

- STYLE:
Answer like a real person, not like a textbook. It's okay to be uncertain or slightly inconsistent.
"""

persona_history = [{"role": "system", "content": PERSONA_SYSTEM_PROMPT}]


model = init_chat_model(model="nvidia/nemotron-3-super-120b-a12b:free", model_provider="openrouter", api_key="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")

checkpointer = MemorySaver()


persona_history = [{"role": "system", "content": PERSONA_SYSTEM_PROMPT}]

def ask_persona(question): 
    persona_history.append({"role": "user", "content":question})

    response = model.invoke(persona_history)
    answer = response.content

    persona_history.append({"role": "assistant", "content": answer})
    return answer


