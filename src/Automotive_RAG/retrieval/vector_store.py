"""
Vector store: takes the chunks we built (step 2) and the embedding model
(step 3), embeds every chunk, and saves it all to a folder on disk using
Chroma.

Two functions, matching the two things you actually do with a vector store:

  build_vector_store()  -- run this ONCE to create it (embeds everything,
                            saves to disk). Re-run only if your chunks or
                            embedding model change.

  load_vector_store()   -- run this every time your app starts, to load
                            the already-built store back from disk instead
                            of re-embedding.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from Automotive_RAG.embeddings.ollama_embedder import get_embedding_model
from Automotive_RAG.chunking.md_splitter import chunk_by_headers

PERSIST_DIR = "data/vectorstore"


def build_vector_store(chunks: list[Document], embedding_model, collection_name: str, persist_dir: str) -> Chroma:

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    return store


def load_vector_store(collection_name: str, embedding_model: str, persist_dir: str) -> Chroma:

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir,
    )
    return store


if __name__ == "__main__":
    # Quick manual check -- build the store from the header-aware chunks,
    # then run one test search against it.
    # Run this yourself locally (needs Ollama running, same as step 3).

    md_path = Path("data/processed/braking_system_requirements.md")
    text = md_path.read_text(encoding="utf-8")
    chunks = chunk_by_headers(text)

    print(f"embedding {len(chunks)} chunks, this may take a few seconds...")
    store = build_vector_store(chunks, collection_name="markdown_header_chunks")
    print(f"saved to {PERSIST_DIR}")

    results = store.similarity_search("What does REQ-EPB-002 require?", k=2)
    print("\n--- top 2 search results ---")
    for r in results:
        print(f"\nmetadata: {r.metadata}")
        print(r.page_content)
