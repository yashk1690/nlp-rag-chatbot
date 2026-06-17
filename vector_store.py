from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownTextSplitter


def build_database(markdown_path: str, collection_name: str):
    print(f"Loading RAG-ready Markdown file: {markdown_path}")

    # 1. Load the document
    loader = TextLoader(markdown_path, encoding="utf-8")
    documents = loader.load()

    # 2. Chunking
    print("Chunking document...")
    text_splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully sliced document into {len(chunks)} chunks.")

    # 3. Waking up Local Embeddings
    print("Waking up local embedding model (FastEmbed)...")
    embeddings = FastEmbedEmbeddings()

    # 4. Save to your existing Qdrant folder
    qdrant_path = "qdrant_db"

    print(f"Loading vectors into Qdrant collection: '{collection_name}'...")

    # --- NEW: Updated class name to QdrantVectorStore ---
    qdrant = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=qdrant_path,
        collection_name=collection_name,
    )

    print(f"\n--- Database Setup Complete ---")
    print(f"Data successfully saved to {qdrant_path}/{collection_name}!")
    return qdrant


if __name__ == "__main__":
    md_file = "data/CV_parsed.md"

    target_collection = "CV"

    build_database(md_file, target_collection)