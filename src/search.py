import os
import logging
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from src.data_loader import load_all_documents

from langchain_core.messages import HumanMessage, AIMessage



load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = 'all-MiniLM-L6-v2', llm_model: str = "openai/gpt-oss-20b", temperature: float = 0.0):
        

        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        data_dir = "data"
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        store_exists = os.path.exists(faiss_path) and os.path.exists(meta_path)
        source_data_is_current = (
            store_exists and self.vectorstore.has_current_source_data(data_dir)
        )

        if source_data_is_current:
            print("[INFO] Source data unchanged. Loading existing vector store.")
            self.vectorstore.load()

        else:
            if store_exists:
                print("[INFO] Source data changed. Rebuilding vector store.")
            else:
                print("[INFO] Vector store not found. Building vector store.")

            docs = load_all_documents(data_dir)
            self.vectorstore.build_from_documents(docs, data_dir)

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY missing in .env file")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=0.0
        )
        print(f"[INFO] Groq LLM initialized: {llm_model} with temperature=0.0")

        self.device_sessions = {}

    def _get_contextualized_query(self, query: str, history) -> str:
        """Pichli baaton ko dekh kar sawal ko mukammal karna (Standalone Query)."""
        if not history:
            return query

        history_text = "\n".join([f"{role}: {msg}" for role, msg in history[-4:]])

        context_prompt = (
            "Given the following chat history and a follow-up question, rephrase the follow-up "
            "into a concise standalone search query. Return ONLY the standalone query text without any "
            "introductory words, preamble, explanations, or quotes. If the question is already standalone, return it as-is.\n\n"
            f"Chat History:\n{history_text}\n\n"
            f"Follow-up Question: {query}\n\n"
            "Standalone Query:"
        )

        response = self.llm.invoke([HumanMessage(content=context_prompt)])
        reformulated = response.content.strip()
        if reformulated.startswith('"') and reformulated.endswith('"'):
            reformulated = reformulated[1:-1].strip()
        return reformulated if reformulated else query

    def search_and_stream(self, query: str, device_id: str):
        try:
            if device_id not in self.device_sessions:
                self.device_sessions[device_id] = []

            history = self.device_sessions[device_id]

            # Quick validation for greetings or trivial single-word fragments
            if not history and len(query.strip().split()) <= 2:
                lower_q = query.strip().lower().rstrip("?!.,")
                if lower_q in ["hi", "hello", "hey", "salam", "assalam o alaikum", "greetings"]:
                    yield "Hello! How can I help you today? You can ask me any question about the documents in the knowledge base."
                    return
                elif lower_q in ["why", "explain", "more", "types", "details", "help"]:
                    yield "Could you please specify which topic or concept you would like to know about?"
                    return

            smart_query = self._get_contextualized_query(query, history)

            results = self.vectorstore.query(smart_query, top_k=10)

            context_blocks = []
            for r in results:
                meta = r.get("metadata") or {}
                txt = meta.get("text", "").strip()
                if not txt:
                    continue
                file_name = meta.get("file_name", "Document")
                page = meta.get("page")
                page_str = f", Page {page + 1}" if page is not None and isinstance(page, int) else ""
                header = f"[{file_name}{page_str}]"
                context_blocks.append(f"--- Source Excerpt {header} ---\n{txt}")

            context = "\n\n".join(context_blocks)

            if not context.strip():
                msg = "I'm sorry, I don't have information on that yet."
                yield msg
                return

            system_instruction = (
                "You are an accurate, highly knowledgeable RAG assistant.\n\n"
                "STRICT INSTRUCTIONS:\n"
                "1. Base your answer strictly and faithfully on the provided context excerpts below.\n"
                "2. Do NOT invent facts, alter events, or fabricate scenes (e.g., do not invent lawsuits, fake escapes, or wrong vessels).\n"
                "3. If a specific detail (such as a name, count, or salary) is NOT mentioned or established in the text, explicitly state that the text does not provide or name it. Do NOT fabricate details or confuse different entities.\n"
                "4. Answer as much of the user's question as is supported by the context. If part of the question is covered and part is unmentioned, address both clearly.\n"
                "5. When presenting factual details from the context, include the relevant document name and page number as citations.\n"
                "6. For analytical questions (e.g. symbolism, framing narrative), provide insightful, balanced analysis grounded strictly in the text's actual events.\n"
                "7. If the question cannot be answered at all from the context, reply: 'I'm sorry, I don't have information on that yet.'\n"
                "8. Keep the tone professional, clear, and articulate.\n\n"
                f"CONTEXT EXCERPTS:\n{context}\n\n"
                f"USER QUESTION: {query}\n\n"
                "ANSWER:"
            )

            full_answer = []
            for chunk in self.llm.stream([HumanMessage(content=system_instruction)]):
                content = chunk.content
                if content:
                    full_answer.append(content)
                    yield content

            complete_answer = "".join(full_answer)
            self.device_sessions[device_id].append(("User", query))
            self.device_sessions[device_id].append(("Assistant", complete_answer))

            if len(self.device_sessions[device_id]) > 50:
                self.device_sessions[device_id] = self.device_sessions[device_id][-50:]
        except Exception as e:
            yield f"\n[Error during generation: {str(e)}]"

    def search_and_summarize(self, query: str, device_id: str) -> str:
        return "".join(list(self.search_and_stream(query, device_id)))
   
        