# 🤖 JARVIS Chatbot Backend - RAG Powered FastAPI

This is the backend for the JARVIS Chatbot, an intelligent assistant built using **FastAPI** and **LangChain**. It uses **Retrieval-Augmented Generation (RAG)** to answer queries based on custom PDF documents.

## 🚀 Key Features
* **PDF Knowledge Integration**: Extracts and processes text from large PDF documents.
* **Vector Search**: Utilizes **FAISS** for efficient semantic search and retrieval.
* **Intelligent LLM**: Powered by **Groq/OpenAI** for natural and context-aware responses.
* **REST API**: Clean FastAPI endpoints for seamless frontend communication.

## 🛠️ Tech Stack
* **Language**: Python 3.10+
* **Framework**: FastAPI
* **RAG Framework**: LangChain
* **Vector Database**: FAISS
* **Environment**: Virtual Environment (venv)

## ⚙️ Setup & Installation

1. **Clone the Repo**:
   ```bash
   git clone [https://github.com/jadranPTI/Langchain-Chatbot-Backend-RAG.git](https://github.com/jadranPTI/Langchain-Chatbot-Backend-RAG.git)
   cd Langchain-Chatbot-Backend-RAG


Create Virtual Environment:
python -m venv venv
.\venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

Environment Variables: Create a .env file and add your API keys:
GROQ_API_KEY=your_api_key_here

Run Server:
uvicorn main:app --reload