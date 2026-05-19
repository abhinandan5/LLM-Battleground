import pandas as pd

def export_chat_history(messages_state):
    """Converts the rich chat history into a downloadable CSV string."""
    export_data = []
    
    for msg in messages_state:
        # We only want to export the assistant's evaluation turns
        if msg["role"] == "assistant" and "eval_data" in msg:
            eval_data = msg["eval_data"]
            
            row = {
                "User_Question": msg.get("question", "N/A"),
                "GPT_Answer": msg.get("gpt_ans", ""),
                "GPT_Accuracy": eval_data.get("Model A", {}).get("Accuracy", ""),
                "GPT_Completeness": eval_data.get("Model A", {}).get("Completeness", ""),
                "Gemini_Answer": msg.get("gemini_ans", ""),
                "Gemini_Accuracy": eval_data.get("Model B", {}).get("Accuracy", ""),
                "Gemini_Completeness": eval_data.get("Model B", {}).get("Completeness", ""),
                "Winner": eval_data.get("Winner", "Tie")
            }
            export_data.append(row)
            
    if not export_data:
        return None
        
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False).encode('utf-8')