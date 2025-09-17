import asyncio
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.events import Event
from typing import Dict, List, Optional

# Configure your API key
GEMINI_API_KEY = "AIzaSyC83fOCAhKrr4iOIf-bWdD9PO4FdJFN8YA"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize ChromaDB
def initialize_chromadb():
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path="./chroma_odoo_skills_db")
    return client.get_collection(name="odoo_hr_skills", embedding_function=embedding_function)

def chroma_query(collection):
    def query_func(query: str, n_results: int = 3, where_filter: Optional[Dict] = None) -> List[Dict]:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas"]
        )
        return [{"document": d, "metadata": m} for d, m in zip(results['documents'][0], results['metadatas'][0])]
    return query_func

def generate_questions(candidate_profile: str, job_requirements: str, conversation_history: str = "") -> str:
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Generate questions based on {candidate_profile}, {job_requirements}, {conversation_history}"
    return model.generate_content(prompt).text

def analyze_response(candidate_profile: str, job_requirements: str, response: str) -> str:
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Analyze response: {response} given profile {candidate_profile} and job {job_requirements}"
    return model.generate_content(prompt).text

def create_agent():
    collection = initialize_chromadb()
    tools = [
        FunctionTool(func=chroma_query(collection)),
        FunctionTool(func=generate_questions),
        FunctionTool(func=analyze_response),
    ]

    instruction = "You are an AI interview assistant."
    return LlmAgent(name="interview-assistant", model="gemini-pro", instruction=instruction, tools=tools)

async def run_agent(agent, user_prompt):
    event_stream = agent.invoke_async(user_prompt)
    final_response = None
    async for event in event_stream:
        if isinstance(event, Event) and event.is_final_response():
            final_response = event.content.parts[0].text if event.content else None
    return final_response

async def main():
    agent = create_agent()
    print("Interview Assistant ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        response = await run_agent(agent, user_input)
        print(f"\nInterview Assistant: {response}")

if __name__ == "__main__":
    asyncio.run(main())
