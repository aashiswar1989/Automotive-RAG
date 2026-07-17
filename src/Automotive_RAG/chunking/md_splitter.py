"""
Header-aware chunking -- the idiomatic LangChain way.

Two steps, run one after another:

STEP 1: Cut the text wherever a '#', '##', or '###' header appears.
        Tool: MarkdownHeaderTextSplitter
        Input:  one big markdown string
        Output: a list of Document objects. Each one remembers which
                headers it was under, e.g. metadata = {"H3": "REQ-BBW-023 ..."}

STEP 2: For any chunk from step 1 that's still too long, cut it further
        by character count.
        Tool: RecursiveCharacterTextSplitter (same tool as the baseline)
        Input:  the list of Documents from step 1
        Output: a list of Documents, further split if needed. LangChain
                copies the H1/H2/H3 metadata onto the new pieces for us
                automatically -- we don't have to do that by hand.
"""

from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Which header levels to cut on, and what to call them in the metadata.
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]


def chunk_by_headers(text: str, max_chars: int = 700, chunk_overlap: int = 50) -> list:
    # Step 1: cut on headers.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep the header text inside the chunk itself
    )
    header_chunks = header_splitter.split_text(text)

    # Step 2: cut anything still too long, by character count.
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=chunk_overlap,
    )
    final_chunks = char_splitter.split_documents(header_chunks)

    return final_chunks


if __name__ == "__main__":
    md_path = Path("data/processed/braking_system_requirements.md")
    text = md_path.read_text(encoding="utf-8")

    chunks = chunk_by_headers(text)

    print(f"chunk count: {len(chunks)}")
    lengths = [len(c.page_content) for c in chunks]
    print(f"avg length: {sum(lengths) / len(lengths):.0f} chars")
    print(f"min/max length: {min(lengths)} / {max(lengths)}")

    for i in [1,6,10,45,60]:
        print("\n--- sample chunk ---")
        print(f"metadata: {chunks[i].metadata}")
        print(chunks[i].page_content)
        print('-'*50)