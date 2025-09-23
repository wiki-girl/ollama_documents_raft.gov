import os
import shutil
import logging
import stat
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# --- CONFIGURATION ---
# The MQA standards document you want to process.
SOURCE_FILE = "academic.staff.txt"
# The directory where the vector database will be stored.
PERSIST_DIR = "chroma_db"
# The name of the collection within the database.
COLLECTION_NAME = "academic_staff_docs"
# The embedding model to use. Must match the one in your verification script.
EMBED_MODEL = "all-MiniLM-L6-v2"

# Set up logging to see the script's progress.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_readonly(func, path, exc_info):
    """
    Error handler for shutil.rmtree.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.
    """
    # Check if the error is an access error
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

def main():
    """
    Main function to process the source file and populate the ChromaDB.
    """
    logging.info("Starting database population process...")

    # --- 1. Clean up existing database (optional) ---
    # If the database directory already exists, we remove it to start fresh.
    # This ensures that any old or outdated information is cleared.
    if os.path.exists(PERSIST_DIR):
        logging.warning(f"Removing existing database at {PERSIST_DIR}")
        # Use the onerror handler to deal with permission issues
        shutil.rmtree(PERSIST_DIR, onerror=remove_readonly)

    # --- 2. Load the source document ---
    try:
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        logging.info(f"Successfully loaded {SOURCE_FILE}")
    except FileNotFoundError:
        logging.error(f"Error: The file '{SOURCE_FILE}' was not found.")
        return

    # --- 3. Split the document into chunks ---
    # We split the text into smaller pieces to make them easier to embed and search.
    # chunk_size: The maximum size of each chunk (in characters).
    # chunk_overlap: The number of characters to overlap between chunks to maintain context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Split the document into {len(chunks)} chunks.")

    # --- 4. Set up the embedding function and ChromaDB client ---
    # This function will be used by ChromaDB to turn our text chunks into vectors.
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    
    # This client saves the database to disk in the PERSIST_DIR directory.
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Get or create the collection. We pass the embedding function here.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    logging.info(f"ChromaDB collection '{COLLECTION_NAME}' is ready.")

    # --- 5. Add the chunks to the database ---
    # We create a unique ID for each chunk before adding it.
    # The collection.add() method handles the embedding and storage automatically.
    if chunks:
        ids = [f"doc_{i}" for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            ids=ids
        )
        logging.info(f"Successfully added {collection.count()} documents to the collection.")
    else:
        logging.warning("No chunks were generated to add to the database.")

    logging.info("✅ Database population process completed successfully!")


if __name__ == "__main__":
    main()