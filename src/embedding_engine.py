import torch
from typing import Union
from transformers import AutoModel, AutoTokenizer

class EmbeddingEngine:
    def __init__(self):
        self.embedding_dim = 384
        self.model = AutoModel.from_pretrained("avsolatorio/NoInstruct-small-Embedding-v0")
        self.tokenizer = AutoTokenizer.from_pretrained("avsolatorio/NoInstruct-small-Embedding-v0")

    def get_doc_embedding(self, text: Union[str, list[str]]):
        self.model.eval()

        if isinstance(text, str):
            text = [text]

        inp = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            output = self.model(**inp)

        # the sentence / document embedding uses the [CLS] representation.
        vectors = output.last_hidden_state[:, 0, :]

        return vectors
    
    def get_query_embedding(self, text: Union[str, list[str]]):
        self.model.eval()

        if isinstance(text, str):
            text = [text]

        inp = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            output = self.model(**inp)

        # The model is optimized to use the mean pooling for queries,
        vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
        vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)

        return vectors