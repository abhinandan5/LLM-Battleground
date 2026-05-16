def validate_output(response_text):
    """
    An MLOps guardrail to validate LLM outputs.
    Returns a tuple: (is_valid: bool, reason: str)
    """
    if not response_text or len(response_text.strip()) < 10:
        return False, "Output too short or empty."
    
    # Common refusal phrases injected by safety filters
    refusals = [
        "as an ai", 
        "i cannot fulfill", 
        "i'm unable to", 
        "i don't have enough information"
    ]
    
    lower_text = response_text.lower()
    for refusal in refusals:
        if refusal in lower_text:
            return False, f"Model refused to answer (Triggered safety/refusal phrase)."
            
    return True, "Valid"