"""
Loads an embedding model. Right now we only support Ollama, running
nomic-embed-text -- but this function is the ONE place that decides
which model to use. Nothing else in the project should hardcode a model
name; everything else just calls get_embedding_model() and gets back a
ready-to-use object.

Why this matters: if you later want to try mxbai-embed-large, or switch
to OpenAI's embedding API, you change it here once, instead of hunting
through every file that touches embeddings.
"""

from langchain_ollama import OllamaEmbeddings


def get_embedding_model(model: str):
    return OllamaEmbeddings(model=model)


if __name__ == "__main__":
    # Quick manual check: embed one sentence, look at the result.
    # Run this yourself locally -- it needs Ollama running with
    # nomic-embed-text pulled (`ollama pull nomic-embed-text`).
    model = get_embedding_model(model = 'nomic-embed-text')
    vector = model.embed_query("What does REQ-BBW-023 require?")
    print(f"vector length: {len(vector)}")
    print(f"first 5 numbers: {vector[:5]}")
