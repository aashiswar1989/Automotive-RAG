"""
The final piece: retrieval + generation, wired together with LCEL.

Four steps, piped in sequence with the '|' operator:

  retriever | prompt | llm | output_parser

  1. retriever      - takes the question, returns the top-k relevant chunks
                       (this is the Chroma vector store from step 4)
  2. prompt         - formats those chunks + the question into one prompt
  3. llm            - the local Llama 3.1 8B model (via Ollama), generates
                       an answer from that prompt
  4. output_parser  - extracts the plain answer text from the LLM's
                       response object

Each step's OUTPUT becomes the next step's INPUT. That's the whole idea.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from Automotive_RAG.retrieval.vector_store import load_vector_store

LLM_MODEL_NAME = "llama3.1:8b"

# Two messages instead of one plain string:
#   - system message: persistent instructions on HOW the model should
#     behave, applies to every question, not tied to any specific query
#   - human message: the actual per-query content -- context + question,
#     filled in fresh every time the chain runs
SYSTEM_PROMPT = """You are a technical assistant for an automotive braking \
system requirements document. Answer questions using ONLY the context \
provided in each message. If the context doesn't contain the answer, say \
you don't know -- do not make anything up or use outside knowledge. When \
relevant, cite the specific REQ-ID(s) your answer is based on."""

HUMAN_PROMPT = """Context:
{context}

Question: {question}"""


def format_chunks(chunks) -> str:
    # Joins the retrieved chunks into one text block, separated by a
    # blank line, so the prompt template's {context} slot gets a single
    # readable string instead of a list of Document objects.
    return "\n\n".join(chunk.page_content for chunk in chunks)


def build_rag_chain(collection_name: str, k: int = 3):
    store = load_vector_store(collection_name)
    retriever = store.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)
    output_parser = StrOutputParser()

    chain = (
        {
            "context": retriever | format_chunks,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | output_parser
    )
    return chain


if __name__ == "__main__":
    # Run this yourself locally -- needs Ollama running with both
    # nomic-embed-text AND llama3.1:8b pulled, and the vector store
    # already built (step 4's script).
    chain = build_rag_chain(collection_name="markdown_header_chunks")

    question = "What happens if the EPB actuator detects sustained motor overcurrent?"
    answer = chain.invoke(question)

    print(f"Q: {question}")
    print(f"A: {answer}")
