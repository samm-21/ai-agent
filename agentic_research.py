import os
import time
from dotenv import load_dotenv
import requests
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import json
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

class ResearchState(TypedDict):
    query: str
    research_data: List[str]
    draft_answer: Optional[str]  # Make draft_answer optional initially
    use_groq_openai: bool

# Load .env variables
load_dotenv()

# API keys and configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_GROQ = os.getenv("USE_GROQ", "true").lower() == "true"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if HUGGINGFACE_TOKEN:
    login(token=HUGGINGFACE_TOKEN)

# === Tavily Agent ===
def web_research_agent(query):
    """Fetch research data using the Tavily API."""
    try:
        headers = {
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
        }
        response = requests.post("https://api.tavily.com/search", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("answer", "") or "\n".join([r['content'] for r in result.get("results", [])])
    except requests.exceptions.RequestException as e:
        print(f"Error during Tavily API request: {e}")
        return "Error: Unable to fetch research data."

# === Research Action for LangGraph ===
def research_action(state: ResearchState):
    """Fetches research data based on the query."""
    query = state['query']
    print("\n[RESEARCHING...]")
    research = web_research_agent(query)
    print("[RESEARCH COMPLETE]")
    print("\n[RESEARCH DATA FETCHED:]\n", research)
    return {"research_data": research}

research_node = RunnableLambda(research_action)

# === Answer Drafting Agent ===
answer_prompt_template = PromptTemplate(
    input_variables=["research_data"],
    template="Based on the following research data, draft a comprehensive answer to the original query:\n\n{research_data}\n\nAnswer:"
)

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)
answer_chain = answer_prompt_template | llm


def answer_action(state: ResearchState):
    """Drafts an answer based on the research data using the local gpt2 model."""
    research_data = state['research_data']
    query = state['query']
    print("\n[DRAFTING ANSWER WITH LOCAL GPT-2 MODEL...]")
    try:
        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        prompt = f"Based on the following research about '{query}', provide a detailed answer:\n\n{research_data}\n\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=10000,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,  # Important for GPT-2
            repetition_penalty=1.2
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("[ANSWER DRAFTED WITH LOCAL GPT-2 MODEL]")
        return {"draft_answer": answer}
    except Exception as e:
        print(f"Error during local GPT-2 inference: {e}")
        return {"draft_answer": "Error: Unable to generate answer using the local GPT-2 model."}

answer_node = RunnableLambda(answer_action)



# === Define LangGraph Workflow ===
workflow = StateGraph(ResearchState)

workflow.add_node("research", research_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("research")
workflow.add_edge("research", "answer")
workflow.add_edge("answer", END)

chain = workflow.compile()

# === Run Example ===
if __name__ == "__main__":
    query = input("Enter your query: ").strip()
    if not query:
        print("Error: Query cannot be empty. Exiting the program.")
    else:
        initial_state = {"query": query, "research_data": [], "draft_answer": None, "use_groq_openai": USE_GROQ}
        result = chain.invoke(initial_state)
        print("\n[FINAL ANSWER]\n", result['draft_answer'])