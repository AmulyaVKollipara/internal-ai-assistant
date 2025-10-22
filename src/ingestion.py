import os
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

curr_directory = os.path.dirname(__file__)
env_file = os.path.join(os.path.dirname(curr_directory), ".env")
pdf_file_path = os.path.join(curr_directory, "pdf_data") 
# file_path = os.path.join(curr_directory, "data")
persistent_directory = os.path.join(curr_directory, "vectorstore")

load_dotenv(env_file)

documents = []

for filename in os.listdir(pdf_file_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_file_path, filename))
            loaded_docs = loader.load()

            # for i, doc in enumerate(loaded_docs):
            #     doc.metadata["source"] = filename
            #     doc.metadata["page_number"] = i + 1  # Page numbers start at 1
            #     doc.metadata["domain"] = "HR" if "HR" in filename.upper() else "IT"

            # documents.extend(loaded_docs)

            # Tag domain based on filename
            domain = "HR" if "HR" in filename.upper() else "IT"
            for doc in loaded_docs:
                doc.metadata["domain"] = domain

            documents.extend(loaded_docs)
            print(f"✅ Loaded and tagged: {filename} as {domain}")


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

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)


    print(f"🔹 Split into {len(split_docs)} chunks. Initializing embeddings...")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
