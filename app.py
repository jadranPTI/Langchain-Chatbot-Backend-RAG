

from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch



if __name__ == "__main__":

    print("[INFO] Bot is starting up...")
    rag_search = RAGSearch()

    print("\n" + "="*50)
    print("  BOT IS READY! (Type 'exit' to stop)")
    print("="*50 + "\n")

    while True:
        
        query = input("You: ").strip()
        device_id = input("device_id: ").strip()

        
        if query and device_id.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Goodbye! Closing the session.")
            break
        
        
        if not query:
            continue

        
        summary = rag_search.search_and_summarize(query, device_id)
        print("Bot:", summary, "\n")



    
















# from src.data_loader import process_all_pdfs
# from src.embedding import EmbeddingPipeline
# from src.vectorstore import VectorStore

# if __name__ == "__main__":
#     docs = process_all_pdfs("data")
#     chunks = EmbeddingPipeline().split_documents(docs)
#     chunkvector = EmbeddingPipeline().generate_embeddings(chunks)
#     vectorstore = VectorStore()

    








