# Bakalauruset-2026

# AI Agent for Cybersecurity Risk Assessment

This repository contains the prototype developed for the bachelor's thesis **"Design and implementation of AI-agent for cybersecurity risk assessment"**.

The prototype implements an AI-agent-based workflow for cybersecurity risk assessment from natural-language scenarios. The workflow consists of several stages: fact extraction, risk identification, risk analysis, and risk evaluation. The system uses large language models and retrieval-augmented generation with E-ITS cybersecurity framework knowledge.

## Workflow

The implemented workflow consists of the following stages:

1. **Fact extraction** — extracts security-relevant facts from the input scenario.
2. **Risk identification** — identifies possible cybersecurity risks based on extracted facts and E-ITS context.
3. **Risk analysis** — assigns likelihood, impact, and risk level.
4. **Risk evaluation** — evaluates risk acceptability and suggests treatment decisions.

## Models Evaluated

The evaluation was conducted using three large language models:

- DeepSeek V3
- Claude Sonnet 4.5
- GPT-5 Mini

Each model was tested on five cumulative cybersecurity scenarios, with five independent runs per scenario.
