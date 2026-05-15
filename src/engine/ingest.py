import os
import sys
import base64
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# 1. Setup the Database Connection
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/vector_db"))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
class VisionImageLoader(BaseLoader):
    """Custom loader that uses a lightweight Vision model to caption images."""
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load(self):
        # Convert image to base64 format required by APIs
        with open(self.file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Initialize the highly cost-effective mini model
        print("Initializing Vision Model (gpt-4o-mini)...")
        chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=300)
        
        # Build the multimodal prompt
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": "Describe this image in high detail. Focus on the main subjects, setting, colors, and any visible text. This description will be saved to a searchable RAG database."
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}
                }
            ]
        )
        
        print("Analyzing image pixels...")
        response = chat.invoke([message])
        
        # Return as a standard LangChain Document so the rest of the pipeline doesn't crash
        return [Document(page_content=response.content, metadata={"source": self.file_path, "type": "image"})]
    
def get_loader(source_path_or_url):
    """Routes the input to the correct LangChain document loader."""
    # Check if it is a YouTube URL
    if source_path_or_url.startswith("http://") or source_path_or_url.startswith("https://"):
        if "youtube.com" in source_path_or_url or "youtu.be" in source_path_or_url:
            print(" Detected YouTube URL. Downloading transcript...")
            return YoutubeLoader.from_youtube_url(source_path_or_url, add_video_info=False,language=["en""en-US", "en-CA", "en-GB"])
        else:
            raise ValueError("Only YouTube URLs are currently supported for web ingestion.")
    
    # Check local file extensions
    ext = os.path.splitext(source_path_or_url)[-1].lower()
    if ext == '.pdf':
        print(f"Detected PDF: {source_path_or_url}")
        return PyPDFLoader(source_path_or_url)
    elif ext == '.txt':
        print(f"Detected Text file: {source_path_or_url}")
        return TextLoader(source_path_or_url)
    elif ext == '.csv':
        print(f" Detected CSV file: {source_path_or_url}")
        return CSVLoader(source_path_or_url)
    elif ext in ['.png', '.jpg', '.jpeg']:
        print(f"Detected Image file: {source_path_or_url}")
        return VisionImageLoader(source_path_or_url)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def ingest_data(source):
    try:
        # 1. Route and Load
        loader = get_loader(source)
        docs = loader.load()
        print(f"Loaded {len(docs)} document pages/sections.")
        
        # 2. Transform (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"Split into {len(splits)} optimized chunks.")
        
        # 3. Load to Persistent Vector Store
        print("Embedding and saving to persistent database...")
        Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=DB_DIR
        )
        print("🎉 Success! The data is now permanently available to your MCP Server.")
        
    except Exception as e:
        print(f" Error during ingestion: {e}")

if __name__ == "__main__":
    print("="*50)
    print("LLm Arena (GPT Vs Gemini)")
    print("="*50)
    
    while True:
        user_input = input("\nEnter a file path, a YouTube URL, or type 'exit' to quit: ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting ingestion engine.")
            sys.exit(0)
            
        if not user_input:
            continue
            
        ingest_data(user_input)