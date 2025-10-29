## Internal AI Assistant  
Developed an AI-powered chatbot assistant that helps employees quickly find accurate answers to HR and IT policy queries by searching internal documents, simplifying the process and improving clarity.

## Tech Stack  
LangChain: Handles the flow of interactions between the AI agent and various data sources.   
Streamlit: Provides the user-friendly web interface for interacting with the chatbot.   
Azure OpenAI: Powers the assistant with GPT models to understand queries and generate responses.   
Chroma: Utilized to store and retrieve data efficiently for HR and IT databases.

## Project Structure  
src  
│── app.py        # Chatbot Application  
│── ingestion.py  # Script for ingestion of data from pdf_data into Chroma database   
│── pdf_data      # Folder containing input documents pertaining to HR and IT  
│── readme.md     # Description and details of the project  

## Key Components  
### Document Ingestion
**Function**: Ingests data from pdf_data and loads into vectorstore Chroma, splitting it into two separate databases - HR and IT

### Chatbot Application  
**Function**: Provides streamlit-based UI to create a RAG Chain that resolves user query based on the input.

## Running the Application
1. Ensure that the input documents are in place.  
Run the ingestion script ingestion.py  
Through this, your documents will be ingested into a Chroma database  
2. Launch the Chatbot  
`streamlit run ./src/app.py` to run the application.


