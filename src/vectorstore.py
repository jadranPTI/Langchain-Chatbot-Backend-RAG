

import os
import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str="faiss_store", embedding_model: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 200):

        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f" [INFO] Loaded embedding model: {embedding_model}")

    @staticmethod
    def source_manifest(data_dir: str) -> list[dict[str, int | str]]:
        data_path = Path(data_dir).resolve()
        supported_extensions = {".pdf", ".txt", ".csv", ".docx", ".doc"}
        files = sorted(
            file_path
            for file_path in data_path.glob("**/*")
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions
        )
        return [
            {
                "path": str(file_path.relative_to(data_path)),
                "size": file_path.stat().st_size,
                "modified_ns": file_path.stat().st_mtime_ns,
            }
            for file_path in files
        ]

    def has_current_source_data(self, data_dir: str) -> bool:
        manifest_path = os.path.join(self.persist_dir, "source_manifest.json")
        if not os.path.exists(manifest_path):
            return False

        with open(manifest_path, "r", encoding="utf-8") as file:
            stored_manifest = json.load(file)

        return stored_manifest == self.source_manifest(data_dir)



    def build_from_documents(self, documents: List[Any], data_dir: str):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        if not documents:
            raise ValueError(f"No supported documents found in {Path(data_dir).resolve()}")

        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save(data_dir)
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")


    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")


    def save(self, data_dir: str):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        manifest_path = os.path.join(self.persist_dir, "source_manifest.json")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(self.source_manifest(data_dir), file, indent=2)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")


    def load(self):
            faiss_path = os.path.join(self.persist_dir, "faiss.index")
            meta_path = os.path.join(self.persist_dir, "metadata.pkl")
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)

            print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")


    def search(self, query_embedding: np.ndarray, top_k: int = 10, threshold: float = 1.5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx != -1 and dist < threshold:
                meta = self.metadata[idx] if idx < len(self.metadata) else None
                results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results
    

    def query(self, query_text: str, top_k: int = 10, threshold: float = 1.5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k, threshold=threshold)
