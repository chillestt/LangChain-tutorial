from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import DeepLake

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()

# instantiate the LLM and embedding models
llm = ChatGoogleGenerativeAI(
    model="gemini-pro-1.0-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.777,
    top_p=0.9,
    top_k = 40
)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "hunter" 
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=gemini_embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)