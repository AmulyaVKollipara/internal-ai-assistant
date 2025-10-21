import os
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv("C:\\Users\\Administrator\\Documents\\Capstone\\.env")

# Define the directory containing the text file and the persistent directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = "C:\\Users\\Administrator\\Documents\\Capstone\\src\\data"
persistent_directory = "C:\\Users\\Administrator\\Documents\\Capstone\\src\\vectorstore"

# Initialize Azure embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
# else:
#     print("All directories found. Proceeding..")

    # Read the text content from the file
    # for file in file_path:
    loader = TextLoader("C:\\Users\\Administrator\\Documents\\Capstone\\src\\data\\hr_policy.txt")
    documents = loader.load()

    loader = TextLoader("C:\\Users\\Administrator\\Documents\\Capstone\\src\\data\\it_policy.txt")
    documents = loader.load()



    print(f"📄 Loaded {len(documents)} document(s). Splitting text into smaller chunks...")

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)

    print(f"🔹 Split into {len(split_docs)} chunks. Initializing embeddings...")

    # Display information about the split documents
    # print("\n--- Document Chunks Information ---")
    # print(f"Number of document chunks: {len(split_docs)}")
    # print(f"Sample chunk:\n{split_docs[0].page_content}\n")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
