from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_gpt_response(context, question, history):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = f"""
    Previous Conversation:
    {history}
    
    Context: 
    {context}
    
    Question: {question}
    
    Answer strictly based on the context and previous conversation:"""
    return llm.invoke(prompt).content

def get_gemini_response(context, question, history):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        max_retries=5
    )
    prompt = f"""
    Previous Conversation:
    {history}
    
    Context: 
    {context}
    
    Question: {question}
    
    Answer strictly based on the context and previous conversation:"""
    return llm.invoke(prompt).content