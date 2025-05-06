# Ai-agent

# ✨ AI agent-based Deep Research
- It is a Deep Research AI Agentic System that crawls websites using Tavily for online information gathering.
- Implements a dual-agent system with one agent focused on research and data collection, while the second agent functions as an answer drafter.
- The system utilizes the LangGraph & LangChain frameworks to effectively organize the gathered information.
- It processes user queries, performs research, and generates detailed answers using a LangGraph workflow. The backend integrates with a React frontend to provide a seamless chatbot experience.

# Features
- 🌍 Real-time web search via Tavily API
- 🤖 Answer generation using **locally-run GPT-2**
- 🧩 Modular design with LangGraph for multi-step flows
- 🔐 Secure API key management with `.env`

# 🛠️ Technologies Used
- Flask: Backend framework for building the API.
- LangGraph: Workflow engine for processing queries.
- Hugging Face Transformers: For local GPT-2 model inference.
- LangChain: Prompt + chain abstraction.
- Tavily API: For fetching research data.
- Python 3.8+
