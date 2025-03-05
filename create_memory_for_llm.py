# Import necessary modules for loading documents, splitting text, and creating embeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load raw PDF(s)
DATA_PATH = 'data/'

def load_pdf_files(data_path):
    """
    Load PDF files from the specified directory using DirectoryLoader.
    
    :param data_path: Path to the directory containing PDF files
    :return: List of loaded documents
    """
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

# Load documents from the specified directory
documents = load_pdf_files(data_path=DATA_PATH)
# Uncomment to print the number of pages and a specific document
print("Length of PDF pages: ", len(documents))
print(documents[10])

# Step 2: Create Chunks
def create_chunks(extracted_data):
    """
    Split the extracted data into smaller chunks for processing.
    
    :param extracted_data: List of documents to be split
    :return: List of text chunks
    """
    # Initialize the text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # Split the documents into chunks
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Create text chunks from the loaded documents
text_chunks = create_chunks(documents)
# Uncomment to print a specific chunk and the total number of chunks
# print(text_chunks[18])
# print(len(text_chunks))

# Step 3: Create vector Embeddings
def get_embedding_model():
    """
    Initialize the HuggingFace embedding model for creating vector embeddings.
    :return: HuggingFaceEmbeddings model
    """
    # Use a pre-trained sentence-transformers model for embedding
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# Get the embedding model
embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

# Create a FAISS vector store from the text chunks and embedding model
db = FAISS.from_documents(text_chunks, embedding_model)

# Save the FAISS vector store locally
db.save_local(DB_FAISS_PATH)