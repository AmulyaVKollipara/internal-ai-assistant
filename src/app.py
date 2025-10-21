import os
import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain



load_dotenv("C:\\Users\\Administrator\\Documents\\Capstone\\.env")

persist_directory = "C:\\Users\\Administrator\\Documents\\Capstone\\src\\vectorstore"


AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# embeddings = AzureOpenAIEmbeddings(
#     model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_API_KEY,
#     openai_api_version=AZURE_OPENAI_API_VERSION
# )

embeddings = AzureOpenAIEmbeddings(
    deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION
)


vectorstore = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings
)


# Create Retrieval-Augmented Generation (RAG) pipeline
retriever = vectorstore.as_retriever()
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    temperature=1
)

# my own code
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "answer only questions relevant to the information "
    "in the database related to HR and IT policies. anything else, please answer by "
    "saying `Sorry, I am unable to answer this as it is beyond my scope.`"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = (
    "You are an assistant for answering HR-related questions. Use ONLY the retrieved context below to answer. "
    "If the context does not contain relevant information, respond with: "
    "`Sorry, I am unable to answer this as it is beyond my scope.`\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def continual_chat():
    print("Start chatting with the AI!" )
    chat_history = []
    while True:
        query = input("You: ")
        if (query.lower() == "exit"):
            break
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result['answer']))

if __name__ == "__main__":
    continual_chat()


# # Streamlit UI
# st.title("🧠 Enterprise Knowledge Assistant")
# st.write("Ask questions about HR policies, IT knowledge, or support documentation.")

# query = st.text_input("🔍 Enter your query:")

# if query:
#     response = qa_chain.run(query)
#     st.write("**AI Response:**", response)