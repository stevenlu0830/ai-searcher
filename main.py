from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# Specify all of the fields that you want as output from your LLM call
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Create a prompt template for the LLM that takes the output of LLM and
# parses it into the ResearchResponse model
parser = PydanticOutputParser(pydantic_object=ResearchResponse)




