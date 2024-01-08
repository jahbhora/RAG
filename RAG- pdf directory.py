# Import necessary modules and components
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader

# Function to set OpenAI API key and initialize ChatOpenAI instance
def initiate_token(api_token):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or api_token
    chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model='gpt-3.5-turbo')
    return chat 

# Set OpenAI API key to an empty string (can be replaced with an actual key)
initiate_token("")

# Function to get a directory loader for PDF files
def get_dir(dir_path):
    source_dir = PyPDFDirectoryLoader(dir_path)
    return source_dir

# Specify the path for the directory containing PDF files
source_dir = get_dir("path here")

# Function to load documents from the specified directory
def load_dir(source_dir):
    loaded_dir = source_dir.load()
    return loaded_dir

# Load documents from the specified directory
loaded_dir = load_dir(source_dir)

# Function to split loaded documents into chunks
def split_text(loaded_dir):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    splits = text_splitter.split_documents(loaded_dir)
    return splits

# Split loaded documents into chunks
splits = split_text(loaded_dir)

# Function to create and persist a Chroma vector store from document chunks
def vec_db(splits, chroma_path):
    vectorDB = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=chroma_path)
    vectorDB.persist()
    print(f"{len(splits)} saved to {chroma_path}")
    return vectorDB

# Create and persist Chroma vector store from document chunks
vectorDB = vec_db(splits, "Chroma_path")

# Function to initialize a RAG (Retrieval-Augmented Generation) chain
def initialize_rag_chain(vectorDB, loaded_dir):
    # Create the retriever
    retriever = vectorDB.as_retriever()

    # Initialize the ChatOpenAI and other components
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    prompt = hub.pull("rlm/rag-prompt")

    # Define the format_docs function
    def format_docs(loaded_dir):
        return "\n\n".join(doc.page_content for doc in loaded_dir)

    # Construct the rag_chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Initialize RAG chain
initialized = initialize_rag_chain(vectorDB, loaded_dir)

# Function to input a prompt to the RAG chain and print the response
def prompt_input(rag_chain, prompt):
    response = rag_chain.invoke(prompt)
    return print(response)

# Input a prompt to the initialized RAG chain and print the response
prompt_input(initialized, "what is risk")
