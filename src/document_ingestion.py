# import os
# from langchain_community.document_loaders import TextLoader
# # from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
# from langchain_openai import AzureOpenAIEmbeddings  # To use Azure OpenAI embeddings through LangChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_experimental.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.schema import Document
# from dotenv import load_dotenv

# # # Load environment variables
# load_dotenv("C:\\Users\\Administrator\\Documents\\Capstone\\.env")
# persist_directory = "C:\\Users\\Administrator\\Documents\\Capstone\\src\\chroma_db"
# file_path = "C:\\Users\\Administrator\\Documents\\Capstone\\src\\data\\hr_policy.txt"

# # Get configurations from .env
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


# if not os.path.exists(persist_directory):
#     print("Persistent directory does not exist. Initializing vector store...")

#     # Ensure the text file exists
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(
#             f"The file {file_path} does not exist. Please check the path."
#         )
# else:
#     print("The directories exist")

# print("🚀 Starting document ingestion...")

# # Load HR policies and IT knowledge base files
# loader = TextLoader(file_path)
# documents = loader.load()

# if not documents:
#     print("❌ No documents found. Please check the file path!")
#     exit()

# print(f"📄 Loaded {len(documents)} document(s). Splitting text into smaller chunks...")

# # Split the document into smaller chunks for better retrieval
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# split_docs = text_splitter.split_documents(documents)

# print(f"🔹 Split into {len(split_docs)} chunks. Initializing embeddings...")

# # Initialize OpenAI embeddings
# embeddings = AzureOpenAIEmbeddings(
#     model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_API_KEY,
#     openai_api_version=AZURE_OPENAI_API_VERSION
# )
# vectorstore = Chroma()
# vectorstore.add_texts(split_docs, AZURE_OPENAI_EMBEDDING_DEPLOYMENT)

# # vectorstore=Chroma.from_documents(
# #     documents=split_docs,
# #     embedding=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
# #     persist_directory=persist_directory)
 

# print("🔄 Converting text into vector embeddings and storing in Chroma...")

# # Convert text chunks into embeddings and store them
# vectorstore.add_documents(split_docs)

# print("✅ Documents successfully added to the enterprise knowledge base in Chroma!")