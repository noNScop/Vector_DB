import pymupdf
import os
from embedding_engine import *
from vector_db import VectorDB
from metadata import Metadata

def read_document(doc, chunks, doc_name):
    chunk_size = 500

    for page_num, page in enumerate(doc): # iterate the document pages
        text = page.get_text() # get plain text
        start_idx = 0

        while start_idx < len(text):
            chunk_text = text[start_idx:start_idx + chunk_size]
            # save chunk metadata
            chunks.append({
                "doc_name": doc_name,
                "page": page_num+1,
                "text": chunk_text.strip(),
            })
            start_idx += chunk_size - 100 # 100 chars overlap

    return chunks

def main():
    while True:
        print()
        print("Select engine:")
        print("[1. NoInstruct-small-Embedding-v0 (small)]")
        print("[2. stella_en_400M_v5 (big)]")
        print()

        choice = input()

        try:
            choice = int(choice)
        except:
            print("Invalid input")
            continue

        if choice == 1:
            engine = NoInstructSmallV0()
            vector_db = VectorDB(engine.embedding_dim, "data/vector_db_NoInstruct")
            metadata = Metadata("data/metadata_NoInstruct.json")
            break

        elif choice == 2:
            engine = Stella400MV5()
            vector_db = VectorDB(engine.embedding_dim, "data/vector_db_Stella")
            metadata = Metadata("data/metadata_Stella.json")
            break

        else:
            print("Wrong choice.")

    while True:
        print()
        print("Select an option:")
        print("[1. Add document to database]")
        print("[2. Add all documents from the directory to database]")
        print("[3. Query the database]")
        print("[4. Clear the database]")
        print("[5. Save the database to files]")
        print("[6. Remove the database files]")
        print("[7. Exit]")
        print()
        
        choice = input()

        try:
            choice = int(choice)
        except:
            print("Invalid input")
            continue

        if choice == 1:
            doc_path = input("PDF document path: ")
            doc_name = os.path.basename(doc_path)

            if not os.path.exists(doc_path):
                print("Invalid document path.")
                continue

            if not doc_path.lower().endswith('.pdf'):
                print("The file is not a PDF document.")
                continue

            try:
                doc = pymupdf.open(doc_path)
            except Exception as e:
                print(f"Error opening document: {e}")
                continue
            
            chunks = []
            read_document(doc, chunks, doc_name)
            doc.close()

            idx = 0
            batch_size = 1024
            while len(chunks) > idx:
                texts = []
                for i in range(idx, min(idx+batch_size, len(chunks))):
                    texts.append(chunks[i]["text"])
                    metadata.add(chunks[i])

                embeddings = engine.get_doc_embedding(texts)
                vector_db.add(embeddings)

                idx += batch_size

            print(f"Succesfully added {doc_name} to the database.")
        elif choice == 2:
            dir_path = input("directory with PDFs path: ")
            if not os.path.exists(dir_path):
                print("Invalid directory path.")
                continue

            pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]

            if not pdf_files:
                print("No PDF documents found in the directory.")
                continue
            
            # Sort pdf files for clarity
            pdf_files.sort()
            for pdf_file in pdf_files:
                doc_path = os.path.join(dir_path, pdf_file)
                doc_name = os.path.basename(doc_path)

                try:
                    doc = pymupdf.open(doc_path)
                except Exception as e:
                    print(f"Error opening document {doc_name}: {e}")
                    continue

                chunks = []
                read_document(doc, chunks, doc_name)
                doc.close()

                idx = 0
                batch_size = 1024
                while len(chunks) > idx:
                    texts = []
                    for i in range(idx, min(idx+batch_size, len(chunks))):
                        texts.append(chunks[i]["text"])
                        metadata.add(chunks[i])

                    embeddings = engine.get_doc_embedding(texts)
                    vector_db.add(embeddings)

                    idx += batch_size

                print(f"Succesfully added {doc_name} to the database.")
        elif choice == 3:
            if vector_db.is_empty():
                print("Database is empty, add document first.")
                continue
            
            print("Query the database (press Enter on an empty line to finish): ")
            query = []
            while True:
                inp = input()
                if not inp:
                    break

                query.append(inp)

            query = "".join(query)
            vector = engine.get_query_embedding(query)
            output = vector_db.search(vector)
            indices = output[1][0]

            for rank, idx in enumerate(indices):
                print()
                print(f"{rank+1}. Similarity: {output[0][0][rank]}")
                print(f"Document name: {metadata.metadata[idx]["doc_name"]}")
                print(f"Page number: {metadata.metadata[idx]["page"]}")
                if rank == 0:
                    print(metadata.metadata[idx]["text"])
                print()

        elif choice == 4:
            metadata.clear()
            vector_db.new_index(engine.embedding_dim)

        elif choice == 5:
            vector_db.save()
            metadata.save()

        elif choice == 6:
            if os.path.exists(vector_db.index_path):
                os.remove(vector_db.index_path)
                print("Vectors file removed.")
            else:
                print("Couldn't find vectors file.")

            if os.path.exists(metadata.metadata_path):
                os.remove(metadata.metadata_path)
                print("Metadata file removed.")
            else:
                print("Couldn't find metadata file.")

            if os.path.exists("data") and not os.listdir("data"):
                os.rmdir("data")

        elif choice == 7:
            break;

        else:
            print("Invalid choice")

if __name__ == '__main__':
    main()