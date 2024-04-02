from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"]
)
question = "What is the capital city of France"

