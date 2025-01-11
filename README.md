# Vector Database Project

This project is a simple and efficient tool for creating and querying a vector database. It leverages pre-trained models to generate embeddings from PDF documents and performs similarity search using the **Faiss** library. The project supports adding new documents, querying for similar content, and managing the database.

## Features

- **Supports two pre-trained embedding models**:
  1. `NoInstruct-small-Embedding-v0` (smaller and faster, suitable for lightweight applications).
  2. `stella_en_400M_v5` (larger model for better embeddings).
- **Efficient text chunking**: Text from PDFs is chunked for processing, ensuring overlapping to preserve context.
- **Query support**: Retrieve relevant document sections with similarity ranking.
- **Persistence**: Save and load vector database and metadata for reusability.
- **Database management**: Add documents, query content, clear the database, or delete saved data files.

## Usage

1. **Select Engine**: Choose between the two supported embedding models.
2. **Choose Action**:
    - **Add a document to the database**: Provide the path to a PDF file, and the project will extract and store its content in the database.
    - **Query the database**: Input a query, and the tool will rank similar chunks of text from the database.
    - **Clear the database**: Remove all stored data from the memory.
    - **Save database to files**: Persist the vector index and metadata to files for later use.
    - **Remove database files**: Delete saved index and metadata files from storage.

### Example workflow

1. **Add a PDF**:
    - Input the PDF file’s path.
    - Text will be extracted, chunked, and embedded into the vector database.
2. **Run a query**:
    - Input a natural language query.
    - The tool retrieves and ranks similar text from stored documents.

4. Save or clear the database as needed.

### Setup

To set up the project locally, follow these steps:

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Vector_DB
```

2. **Download Faiss**: Installing faiss is not easy, you can start by folowing their [installation guide](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss), if this guide is not enough for you to get Faiss working (it wasn't enough for me), take a look at a list of commands I had to run on MacOS to install it, feel free to get inspired:
```bash
brew install libomp
brew install swig
brew install gflags

pip install numpy
git clone https://github.com/facebookresearch/faiss.git
cd faiss
git checkout fix_nightly_build

cmake -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DPython_EXECUTABLE=$(which python3) \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenMP_CXX_FLAGS="-I$(brew --prefix libomp)/include" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib" \
    -DSWIG_EXECUTABLE=$(which swig) \
    -DOpenMP_C_FLAGS="-I$(brew --prefix libomp)/include" \
    -DOpenMP_C_LIB_NAMES="omp" \
    -DCMAKE_PREFIX_PATH=$(brew --prefix gflags)

cd build
make -j7
cd ..
(cd build/faiss/python/ ; python3 setup.py build)

pip install faiss-cpu
```

3. **Install remaining libraries**:
```bash
pip install -r requirements.txt
```

4. **Run the app**: To run the application, simply execute:

`python main.py`

## Project Structure
Here’s a quick overview of the project structure:
```bash
Vector_DB/
├── data/              # Directory for saving database files
├── src/               # Source code
└── README.md          # Project description
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.