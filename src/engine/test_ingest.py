from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

# Point to the exact same permanent database our server uses
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/vector_db"))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Let's create some dummy data
sample_docs = [
    Document(page_content="Testing Log 001: The Enterprise RAG Arena architecture is currently online and functioning perfectly."),
    Document(page_content="Fun Fact: The largest planet in the solar system is Jupiter, but the Arena database is growing faster!"),
    Document(page_content="System Admin Profile: Srajan Goyal successfully integrated the Model Context Protocol (MCP) on May 5th, 2026.")
]

print("Injecting data into persistent database...")

# This loads the data and saves it permanently to your hard drive
vectorstore = Chroma.from_documents(
    documents=sample_docs, 
    embedding=embeddings, 
    persist_directory=DB_DIR
)

print("Data successfully saved! You can now ask Claude about it.")