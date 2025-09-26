import json
from time import perf_counter
from core_api import Grok

def load_total_prompt(filename: str = "total_prompt.json"):
    """
    Load the total_prompt list from a JSON file.
    
    Args:
        filename (str): File to read the list from.
    
    Returns:
        List: The list of prompts retrieved from the file.
    """
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ File {filename} not found. Returning empty list.")
        return []

def json_test(filename, rag_status):
    total_prompt = load_total_prompt(filename)
    print(f"The length of total prompt [{rag_status}] is ", len(total_prompt))
    t0 = perf_counter()
    try:
        response = Grok.send_text(total_prompt, stream=True, grok_model="grok-2-1212")
    except Exception as e:
        print("grok failed ", e)
    # response = Claude.process_text(messages, system_dict, stream=True)

    llm_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            llm_response += content
    t1 = perf_counter()

    grok_time = (t1 - t0) * 1000
    print(f"This is the grok time for [{rag_status}] ", grok_time)

if __name__ == "__main__":
    json_test("/workspace/non_rag_total_prompt.json", "non_RAG")
    json_test("/workspace/rag_total_prompt.json", "RAG")



