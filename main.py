from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

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

# Create a chat prompt template that includes the system message, user query, 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            You are a helpful research assistant that will help generate a research report.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text \n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Create the agent that will use the LLM and the prompt template
tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Create an AgentExecutor that will run the agent with the provided tools
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
query = input("Enter your research query: ")
raw_response = agent_executor.invoke({"query": "What is the impact of climate change on polar bear populations?"})
print(raw_response)

# Parse the raw response into the structured format using the PydanticOutputParser
try:
    structured_response = parser.parse(raw_response["output"][0]["text"])
except Exception as e:
    print("Error parsing response", e, "Raw response - ", raw_response)