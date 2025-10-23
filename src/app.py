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
from langchain.chains import LLMChain

curr_directory = os.path.dirname(__file__)
env_file = os.path.join(os.path.dirname(curr_directory), ".env")
persist_directory = os.path.join(curr_directory, "vectorstore")
hr_directory = os.path.join(persist_directory, "HR")
it_directory = os.path.join(persist_directory, "IT")


load_dotenv(env_file)


AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
TEXT_EMBEDDING_DEPLOYMENT = os.getenv("TEXT_EMBEDDING_DEPLOYMENT")
TEXT_EMBEDDING_VERSION = os.getenv("TEXT_EMBEDDING_VERSION")
TEXT_EMBEDDING_URL = os.getenv("TEXT_EMBEDDING_URL")
TEXT_EMBEDDING_KEY = os.getenv("TEXT_EMBEDDING_KEY")


embeddings = AzureOpenAIEmbeddings(
    deployment=TEXT_EMBEDDING_DEPLOYMENT,
    azure_endpoint=TEXT_EMBEDDING_URL,
    api_key=TEXT_EMBEDDING_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION
)


hr_vectorstore = Chroma(persist_directory=hr_directory, embedding_function=embeddings)
it_vectorstore = Chroma(persist_directory=it_directory, embedding_function=embeddings)

hr_retriever = hr_vectorstore.as_retriever()
it_retriever = it_vectorstore.as_retriever()


# Create Retrieval-Augmented Generation (RAG) pipeline
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    temperature=1
)


system_prompt = (
    "Given a chat history and the latest user question "
    "answer only questions relevant to the information "
    "in the database related to HR and IT policies. "
    "Categorize the question based on IT or HR."
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

# Creating classifier for HR or IT
classification_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a classifier that determines whether a user's question is about HR, IT, or neither.\n"
     "Respond with only one word: 'HR' or 'IT'\n"
     "If the question refers to an older question or chat history then refer to the history to determine 'HR' or 'IT'."
     "Examples:\n"
     "- 'How do I reset my password?' → IT\n"
     "- 'What is the leave policy?' → HR\n"
     "- 'What's my roles and responsibilites?' → HR"),
    ("human", "{input}")
])

classifier_chain = LLMChain(llm=llm, prompt=classification_prompt)

def classify_query(user_input: str) -> str:
    result = classifier_chain.invoke({"input": user_input})
    return result["text"].strip().upper()



final_prompt = (
    "You are an assistant for answering IT and HR-related questions.\n"
    "Use the retrieved context and the chat history to answer the user's question.\n"
    "If the user is referring to a previous question or answer, use the chat history to respond appropriately.\n"
    "If the query is not relevant to IT or HR policies and cannot be answered from context or history, respond with:\n"
    "`Sorry, I am unable to answer this as it is beyond my scope.`\n\n"
    "Only mention source and other details if you are fetching relevant details.\n"
    "If source is from HR then begin with `Database: HR`, if IT then begin with `Database: IT`.\n"
    "Use ONLY the retrieved context and chat history to answer.\n"
    "If mentioning policy, it has to be exact. When summarizing you can elaborate.\n"
    "For each answer, include the source filename, title, point number (if present) from the metadata.\n"
    "Do not answer more than 3 lines.\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", final_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)  


question_answer_chain = create_stuff_documents_chain(llm, prompt_template)


 
# Initialize chat history in Streamlit session state

st.title("Nova Internal Knowledge Base Assistant")
st.markdown("Please ask your question below!")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_ended" not in st.session_state:
    st.session_state.session_ended = False

# Stop interaction if session has ended
if st.session_state.session_ended:
    st.markdown("The session has ended. Please refresh the page to start a new one.")
    st.stop()


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

    domain = classify_query(user_input)
    if domain == "HR":
        retriever = hr_retriever
    else:
        retriever = it_retriever
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, system_prompt_template
    )
    

    retrieved_docs = history_aware_retriever.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })

    # Step 4: Fallback to keyword search if retrieval is weak
    if not retrieved_docs or len(retrieved_docs) < 2:
        fallback_docs = retriever.similarity_search(user_input, k=4)
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(lambda _: {"context": fallback_docs}, question_answer_chain)
    else:
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    if user_input.strip().lower() == "exit":
        st.session_state.session_ended = True
        with st.chat_message("assistant"):
            st.markdown("You've ended the sesssion. Please refresh the page to start a new one.")
        st.stop()

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