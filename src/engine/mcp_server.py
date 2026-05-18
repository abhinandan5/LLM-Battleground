from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# 1. Initialize the Server
mcp = FastMCP("Arena_RAG_Server")

# 2. Setup the persistent database connection
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/vector_db"))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    Search the Arena's Vector Database for context.
    Use this tool whenever the user asks a question about the uploaded documents.
    """
    try:
        # Connect to the persistent database
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
        # Search for chunks
        docs = vectorstore.similarity_search(query, k=top_k)
        
        if not docs:
            return "No relevant information found in the knowledge base."
            
        # Format the results
        context = "\n\n".join([f"Chunk {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
        return context
        
    except Exception as e:
        return f"Error accessing the knowledge base: {str(e)}"

if __name__ == "__main__":
    print("Starting Arena MCP Server...")
    mcp.run()