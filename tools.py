from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

# Wikipedia tool for querying information
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

# Wikipedia API wrapper: This tool retrieves information from Wikipedia
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

# Wikipedia tool for querying information using the API wrapper
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)