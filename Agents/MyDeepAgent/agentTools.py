from typing import Annotated, List, Literal, Union
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from dotenv import load_dotenv
load_dotenv()


@tool 
def ask_user(question: str) -> str: 
    """ Ask the user a question and wait for their response.
    Use this every time you need to input from the user during the interview  
    """
    return question