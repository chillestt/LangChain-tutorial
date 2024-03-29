from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os


def init():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

init()

## 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro-latest",
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.777,
    top_p=0.9,
    top_k=32
)
prompt = PromptTemplate.from_template(
    "You are an expert in language learning. Now help me to improve my language skills by providing insightful answer for my question: {question}."
)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

conversation.predict(input="Tell me about yourself")
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")
# prompt_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
# text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."

# question = "I have no experience about Mandarine before. Show me the most effective way to study Mandarine to HSK4 in 5 months"
# print(prompt_chain.run(question=question))