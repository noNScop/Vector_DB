import json
import os

# Each entry in the metadata is a dictionary with following entries:
# "doc_name": document name
# "page": page number in the document,
# "text": text associated with it,
class Metadata:
    def __init__(self, metadata_path: str = "data/metadata.json"):
        self.metadata_path = metadata_path
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

    def add(self, chunk):
        self.metadata.append(chunk)

    def save(self):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)

    def clear(self):
        self.metadata = []
