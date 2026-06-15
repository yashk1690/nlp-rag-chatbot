import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant


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
    qdrant_path = "qdrant_db"  # Pointing to your existing folder

    print(f"Loading vectors into Qdrant collection: '{collection_name}'...")

    # LangChain's Qdrant integration is smart:
    # If the collection doesn't exist, it creates it.
    # If it DOES exist (like in your mega-database scenario), it simply appends the new chunks to it!
    qdrant = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=qdrant_path,
        collection_name=collection_name,
    )

    print(f"\n--- Database Setup Complete ---")
    print(f"Data successfully saved to {qdrant_path}/{collection_name}!")
    return qdrant


if __name__ == "__main__":
    md_file = "data/sample_report_2_parsed.md"

    # If you change this name to "mega_course_library" and run this script
    # on 10 different files, they will all funnel into the same massive database!
    target_collection = "sample_report_2"

    build_database(md_file, target_collection)