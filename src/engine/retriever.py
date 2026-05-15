import os
from PIL import Image
import base64 # <-- NEW IMPORT
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, YoutubeLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_image(image_path):
    """Uses Gemini Vision to convert an image into a highly detailed text document."""
    
    # 1. Convert the local image into a Base64 string that the API can read
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    # Get the file extension (jpg, png, etc) to tell the API what type of image it is
    ext = os.path.splitext(image_path)[-1].lower().replace('.', '')
    image_data = f"data:image/{ext};base64,{encoded_string}"

    vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    
    prompt = """
    Analyze this image in extreme detail. 
    1. If there is text, transcribe it exactly.
    2. If it is a chart or graph, summarize the data points and trends.
    3. If it is a photo or diagram, describe the subjects, relationships, and context thoroughly.
    Your output will be used in a search database, so be highly descriptive.
    """
    
    # 2. Pass the Base64 image data to the model
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data}}
        ]
    )
    
    response = vision_model.invoke([message])
    return [Document(page_content=response.content, metadata={"source": image_path})]

def get_loader(source_path_or_url):
    """Routes the input to the correct extraction logic."""
    if source_path_or_url.startswith("http://") or source_path_or_url.startswith("https://"):
        if "youtube.com" in source_path_or_url or "youtu.be" in source_path_or_url:
            return YoutubeLoader.from_youtube_url(source_path_or_url, add_video_info=True)
        else:
            raise ValueError("Only YouTube URLs are currently supported for web ingestion.")
    
    ext = os.path.splitext(source_path_or_url)[-1].lower()
    if ext == '.pdf':
        return PyPDFLoader(source_path_or_url)
    elif ext == '.txt':
        return TextLoader(source_path_or_url)
    elif ext == '.csv':
        return CSVLoader(source_path_or_url)
    elif ext in ['.png', '.jpg', '.jpeg']:
        # Return a custom class-like object that has a .load() method
        class CustomImageLoader:
            def load(self):
                return process_image(source_path_or_url)
        return CustomImageLoader()
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def process_document(source_path_or_url):
    # 1. Dynamic Extraction (Now supports images!)
    loader = get_loader(source_path_or_url)
    docs = loader.load()
    
    # 2. Transformation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 3. Load to Vector Store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # 4. Initialize Re-ranking Pipeline
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever