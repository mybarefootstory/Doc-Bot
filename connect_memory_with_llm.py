# Import necessary modules for setting up the language model, prompts, and vector store
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Step 1: Load environment variables
# Load environment variables from a .env file to access sensitive information like tokens
load_dotenv()
HF_TOKEN = os.environ.get('HF_TOKEN')  # Hugging Face API token
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Model repository ID

def load_llm(huggingface_repo_id):
    """
    Initialize the Hugging Face language model endpoint with specified parameters.
    
    :param huggingface_repo_id: The repository ID of the model to load
    :return: An instance of HuggingFaceEndpoint configured with the model
    """
    # Create an instance of HuggingFaceEndpoint with model parameters
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,  # Controls the randomness of the model's output
        model_kwargs={
            "tokens": HF_TOKEN,  # Authentication token for accessing the model
            "max_length": 512  # Maximum length of the generated output
        }
    )
    return llm

# Step 2: Define a custom prompt template for the QA system
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context. Just answer that the answer for the asked query is out of your present capabilities.

Context: {context}
Question: {question}

Start the answer directly. No small talk please
"""

def set_custom_prompt(custom_prompt_template):
    """
    Create a prompt template for the QA system.
    
    :param custom_prompt_template: The template string for the prompt
    :return: A PromptTemplate object configured with the template
    """
    # Initialize a PromptTemplate with the given template and input variables
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 3: Load the FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"  # Path to the local FAISS database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Embedding model
# Load the FAISS vector store with the embedding model
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create a QA chain using the language model and vector store
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),  # Load the language model
    chain_type="stuff",  # Type of chain to use
    retriever=db.as_retriever(search_kwargs={'k': 3}),  # Configure the retriever with top-k results
    return_source_documents=True,  # Return source documents with the response
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}  # Use the custom prompt
)

# Step 5: Invoke the QA chain with a user query
user_query = input("Write query here: ")  # Prompt the user for a query
response = qa_chain.invoke({'query': user_query})  # Invoke the QA chain with the query
# print("Result: ", response["result"])  # Print the result of the query
# print("Source Documents: ", response["source_documents"])  # Print the source documents used