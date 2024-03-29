from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent
from dotenv import load_dotenv
from langchain.agents import Tool
# except from GoogleAPIWrapper, everythng is ok and can be imported from langchain agents
from langchain_community.utilities import GoogleSearchAPIWrapper

from langchain.chains import LLMChain

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-1.0-pro-latest",
    temperature=0.777
)

search = GoogleSearchAPIWrapper()

prompt = PromptTemplate(
    input_variables=['query'],
    template="Write a summary of the following text: {query}"
)
summarizer = LLMChain(llm=llm, prompt=prompt)
tools = [
    Tool(
        name="summarizer",
        func=summarizer.run,
        description="useful for text summarization"
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iteration=6
)

response = agent("Who is the richest person alive at the moment? and summarize his life achievements")
print(response['output'])