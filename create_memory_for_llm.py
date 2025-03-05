from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load raw PDF(s)
DATA_PATH = 'data/'

def load_pdf_files(data_path):
    # Construct the full path to the PDF files
    full_path = os.path.join(os.getcwd(), data_path)

    # Initialize the DirectoryLoader with the constructed path
    loader = DirectoryLoader(
        full_path,
        glob='*.pdf',  # Specify to load only PDF files
        loader_cls=PyPDFLoader  # Use PyPDFLoader for each PDF file
    )

    # Load the documents
    documents = loader.load()
    return documents

documents = load_pdf_files(data_path=DATA_PATH)
# print("Length of PDF pages: ", len(documents))
# print(documents[10])


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
# print(text_chunks[18])
# print(len(text_chunks))


# Step 3: Create vector Embeddings

def get_embedding_model():
    """
    all-MiniLM-L6-v2
    This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
   """
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()


# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)










