import faiss
import os

class VectorDB:
    def __init__(self, embedding_dim: int, index_path: str):
        self.index_path = index_path
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.new_index(embedding_dim)

    def add(self, embeddings):
        self.index.add(embeddings)

    def search(self, query_vector, k: int = 5):
        return self.index.search(query_vector, k)

    def save(self):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

    def new_index(self, embedding_dim):
        self.index = faiss.IndexFlatIP(embedding_dim)

    def is_empty(self):
        return not self.index.ntotal