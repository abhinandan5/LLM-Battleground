import json
import re

def extract_json(response_text):
    """Safely extracts JSON from an LLM, handling markdown and control characters."""
    try:
        # 1. Violently strip any markdown code blocks (even if they are squished)
        text = re.sub(r'^```(?:json)?\s*', '', response_text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        
        # 2. Find the absolute bounds of the JSON object
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            clean_json = text[start:end+1]
            
            # 3. strict=False is the magic key! It ignores weird LLM whitespace/tabs
            return json.loads(clean_json, strict=False)
        else:
            print("Error: No JSON brackets found.")
            return None
            
    except json.JSONDecodeError as e:
        # Now we can see exactly why it failed in the terminal!
        print(f"JSON Parsing Error: {e}\nRaw String: {clean_json}")
        return None