from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def evaluate_responses(context, question, gpt_answer, gemini_answer, judge_choice="Claude 3 Haiku"):
    """Evaluates the outputs using the dynamically selected LLM Judge."""
    
    # 1. Route to the correct model based on UI selection
    if judge_choice == "Claude 4.6 Sonnet (Best)":
        llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    elif judge_choice == "GPT-4o (OpenAI)":
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    else:
        # Default fallback
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0) 
    
    # 2. The Universal Judge Prompt
    judge_prompt = f"""
    You are an impartial expert judge. You will be provided with a source context, a user question, and two AI-generated answers.
    Your job is to evaluate both answers based on the context.
    
    Context: {context}
    Question: {question}
    
    Model A (GPT): {gpt_answer}
    Model B (Gemini): {gemini_answer}
    
    Evaluate both on a scale of 1-10 for:
    1. Accuracy (Does it state facts from the context?)
    2. Completeness (Does it answer the whole question?)
    
    Respond STRICTLY in the following JSON format, nothing else:
    {{
        "Model A": {{"Accuracy": score, "Completeness": score, "Feedback": "short text"}},
        "Model B": {{"Accuracy": score, "Completeness": score, "Feedback": "short text"}},
        "Winner": "Model A or Model B or Tie"
    }}
    
    CRITICAL: Do NOT wrap your response in markdown code blocks (no ```json). Output ONLY the raw JSON brackets and content.
    """
    
    # max_retries handles minor network blips automatically!
    return llm.with_config(max_retries=3).invoke(judge_prompt).content