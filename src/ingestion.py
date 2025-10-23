import os
from dotenv import load_dotenv
import warnings

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader


# Setting paths and variables to be used
curr_directory = os.path.dirname(__file__)
env_file = os.path.join(os.path.dirname(curr_directory), ".env")
pdf_file_path = os.path.join(curr_directory, "pdf_data") 
persistent_directory = os.path.join(curr_directory, "vectorstore")
hr_directory = os.path.join(persistent_directory, "HR")
it_directory = os.path.join(persistent_directory, "IT")

load_dotenv(env_file)

documents = []
hr_docs = []
it_docs = []

for filename in os.listdir(pdf_file_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_file_path, filename))
            loaded_docs = loader.load()

            if "HR" in filename.upper():
                 domain_name = "HR"
                 hr_docs.extend(loaded_docs)
            if "IT" in filename.upper():
                 domain_name = "IT"
                 it_docs.extend(loaded_docs) 

            print(f"Loaded {filename} into {domain_name}")


# Initialize Azure embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment=os.getenv("TEXT_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("TEXT_EMBEDDING_VERSION"),
    azure_endpoint=os.getenv("TEXT_EMBEDDING_URL"),
    api_key=os.getenv("TEXT_EMBEDDING_KEY")
)

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    print(f"📄 Loaded {len(documents)} document(s). Splitting text into smaller chunks...")

    warnings.filterwarnings("ignore")
    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separator="\n")
    hr_chunks = text_splitter.split_documents(hr_docs)
    it_chunks = text_splitter.split_documents(it_docs)

    print(f"🔹 Split into HR: {len(hr_chunks)} chunks and IT: {len(it_chunks)} chunks.")

    # Create the vector store and persist it automatically
    print("\nCreating vector store for HR and IT....")
    hr_db = Chroma.from_documents(hr_chunks, embeddings, persist_directory=hr_directory)
    it_db = Chroma.from_documents(it_chunks, embeddings, persist_directory=it_directory)

    print("\nFinished creating vector store!")

else:
    print("Vector store already exists. No need to initialize.")
