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

curr_directory = os.path.dirname(__file__)
env_file = os.path.join(os.path.dirname(curr_directory), ".env")

load_dotenv(env_file)

persist_directory = os.path.join(curr_directory, "vectorstore")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

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

# 
system_prompt = (
    "Given a chat history and the latest user question "
    "answer only questions relevant to the information "
    "in the database related to HR and IT policies. "
    "Catgorize the question based on IT or HR."
    "Respond to pleasantries by asking "
    "`How may I help you today?`"
)

system_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, system_prompt_template
)

final_system_prompt = (
    "You are an assistant for answering IT and HR-related questions."
    "Use ONLY the retrieved context below to answer. "
    "Mention whether you are retrieving the data from IT or HR in the beginning "
    "For answering IT questions, use the document's metadata labelled IT "
    "and for HR the document's metadata is labelled HR"
    "Do not answer more than 3 lines."
    "If the context does not contain relevant information, respond with: "
    "`Sorry, I am unable to answer this as it is beyond my scope.`\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", final_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

 
# Initialize chat history in Streamlit session state

st.title("🧠 Enterprise Knowledge Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display existing messages
for i in range(0, len(st.session_state.chat_history), 2):
    user_msg = st.session_state.chat_history[i].content
    ai_msg = st.session_state.chat_history[i + 1].content if i + 1 < len(st.session_state.chat_history) else ""
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(ai_msg)

# Chat input
user_input = st.chat_input("Please enter your query...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    result = rag_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })

    with st.chat_message("assistant"):
        st.markdown(result['answer'])

    # Update chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(SystemMessage(content=result['answer']))