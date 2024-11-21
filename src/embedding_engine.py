import torch
from typing import Union
from transformers import AutoModel, AutoTokenizer, logging
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings related to stella model
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")
logging.set_verbosity_error()

class NoInstructSmallV0:
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
    
class Stella400MV5:
    def __init__(self):
        self.embedding_dim = 1024
        self.model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            trust_remote_code=True,
            device="cpu",
            config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
        )

    def get_doc_embedding(self, text: Union[str, list[str]]):
        if isinstance(text, str):
            text = [text]

        return self.model.encode(text, show_progress_bar=True)

    def get_query_embedding(self, text: Union[str, list[str]]):
        if isinstance(text, str):
            text = [text]

        return self.model.encode(text, prompt_name="s2p_query")