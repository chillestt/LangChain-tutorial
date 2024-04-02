from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

template = """
You are a language expert. Your responsibility is to provide in-depth,hands-on tips and tricks, as well as valuable lessons, facts, knowledge to help other people to master language
You will be asked some questions, and answer it thoroughly

# Question: {question}
"""
prompt = PromptTemplate(
    template=template,
    input_variables=['something'],
    output_parser=StrOutputParser()
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro-latest",
    temperature=0.777,
    top_k=32,
    top_p=0.9,
    convert_system_message_to_human=True
)

llmChain = LLMChain(prompt=prompt, llm=llm)
print(llmChain.run("mastering Chinese in 1 year"))

# messages = [
#     SystemMessage(
#         content="You are a language expert"
#     ),
#     HumanMessage(
#         content="The most effective way to learn Chinese Vocabularies?"
#     )
# ]

# print(llm(messages))

