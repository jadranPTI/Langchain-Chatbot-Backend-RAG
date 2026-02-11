import os
import logging
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, AIMessage



load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = 'all-MiniLM-L6-v2', llm_model="llama-3.1-8b-instant"):
        

        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)

        else:
            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY missing in .env file")

        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")


        
        self.device_sessions = {}



    def _get_contextualized_query(self, query, history):
        """Pichli baaton ko dekh kar sawal ko mukammal karna (Custom Logic)"""
        if not history:
            return query

        
        history_text = ""
        for role, msg in history[-4:]:
            history_text += f"{role}: {msg}\n"


        context_prompt = (
            f"Given the following chat history, reformulate the latest user question "
            f"to be a standalone question. If it's already standalone, just return it.\n\n"
            f"History:\n{history_text}\n"
            f"Latest Question: {query}\n"
            f"Standalone Question:"
        )
        
        response = self.llm.invoke([HumanMessage(content=context_prompt)])
        return response.content

  
    def search_and_summarize(self, query: str, device_id: str) -> str:
        
        if device_id not in self.device_sessions:
            self.device_sessions[device_id] = []

        history = self.device_sessions[device_id]

        if not history:
        
            check_prompt = (
                "You are a query validator. A user is starting a NEW conversation. "
                f"User says: '{query}'\n"
                "Is this question complete enough to answer without any previous context? "
                "If it's too vague (like 'types?', 'explain more', 'benefits'), reply with 'VAGUE'. "
                "If it's a clear question, reply with 'CLEAR'."
            )
            validation = self.llm.invoke([HumanMessage(content=check_prompt)]).content
        
            if "VAGUE" in validation.upper():
                return "I'd love to help, but since we just started, I'm not sure what you're referring to. Could you please specify the topic?"

        
        smart_query = self._get_contextualized_query(query, history)

        
        results = self.vectorstore.query(smart_query, top_k=10)
        context = "\n\n".join([r["metadata"].get("text", "") for r in results if r["metadata"]])

        if not context.strip():
            return "Sorry, I don't have knowledge about it."

        
        system_instruction = (
            # "You are a helpful assistant. Answer the user's question efficiently "
            # f"\n\nContext: {context}\n\nQuestion: {query}"
            "You are a highly intelligent and friendly AI assistant."
            "Your goal is to explain concepts in a natural, conversational way. "
            "\n\nSTRICT RULES:\n"
            "1. Do NOT mention page numbers, chapter numbers, figure titles, or 'provided context'.\n"
            "2. Do NOT say 'According to the text' or 'The book says'.\n"
            "3. Speak as if this knowledge is your own expertise.\n"
            "4. If you don't know the answer based on the context, just say: 'I'm sorry, I don't have information on that yet.'\n"
            "5. Keep the tone helpful and professional.\n"
            "6. Only answer based on the knowledge found in the context. \n"
            "7. If the user asks about your identity or what you can do, explain what topics are available in your documents (the context).\n"
            f"\n\nContext for your knowledge: {context}"
            f"\n\nUser's Question: {query}"
        )
        
        response = self.llm.invoke([HumanMessage(content=system_instruction)])
        answer = response.content

        
        self.device_sessions[device_id].append(("User", query))
        self.device_sessions[device_id].append(("Assistant", answer))

       
        if len(self.device_sessions[device_id]) > 50:
            self.device_sessions[device_id] = self.device_sessions[device_id][-50:]

        return answer
   
        