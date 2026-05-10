import json
import re
from typing import Any, Dict


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