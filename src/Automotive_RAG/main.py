from pathlib import Path
from Automotive_RAG.utils.ollama_serve import ensure_ollama_running

from Automotive_RAG.config import ConfigLoader
from Automotive_RAG.ingestion.pdf_to_md import convert
from Automotive_RAG.chunking.md_splitter import chunk_by_headers
from Automotive_RAG.embeddings.ollama_embedder import get_embedding_model
from Automotive_RAG.retrieval.vector_store import build_vector_store, load_vector_store
from Automotive_RAG.chains.rag_chain import build_rag_chain

class RAG_Pipeline:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.vector_store = None
        self.rag_chain = None
        self.embedding_model = get_embedding_model(model = self.config.embedding.model)
    
    def ingestion(self):
        pdf_path = Path(self.config.data.source_pdf)
        md_path = Path(self.config.data.source_md)        
        md_path = convert(pdf_path = pdf_path, md_path = md_path)

        md_text = md_path.read_text(encoding="utf-8")
        chunks = chunk_by_headers(md_text, max_chars=self.config.chunking.chunk_size,
                                    chunk_overlap=self.config.chunking.chunk_overlap)


        self.vector_store = build_vector_store(chunks, self.embedding_model, 
                                                    collection_name=self.config.embedding.collection_name,
                                                    persist_dir = self.config.embedding.persist_dir)
        
    def store_exists(self) -> bool:
        store = load_vector_store(collection_name=self.config.embedding.collection_name,
                                embedding_model=self.embedding_model,
                                persist_dir = self.config.embedding.persist_dir)


        return (True if store._collection.count() > 0 else False)
        
    def get_ragchain(self):

        if self.vector_store is None:
            self.vector_store = load_vector_store(collection_name=self.config.embedding.collection_name,
                                        embedding_model=self.embedding_model,
                                        persist_dir = self.config.embedding.persist_dir)

        retriever = self.vector_store.as_retriever(search_type = self.config.retriever.search_type,
                                                search_kwargs={"k": self.config.retriever.k})

        self.rag_chain = build_rag_chain(model = self.config.llm.model,
                                    temperature = self.config.llm.temperature,
                                    retriever = retriever
                                    )
        return self.rag_chain

    def query(self, question: str):
        if self.rag_chain is  None:
            self.rag_chain = self.get_ragchain()

        return self.rag_chain.invoke(question)

if __name__ == '__main__':
    ensure_ollama_running()
    config = ConfigLoader('configs/settings.yaml')
    rag_pipeline = RAG_Pipeline(config)
    
    if not rag_pipeline.store_exists():
        rag_pipeline.ingestion()

    response = rag_pipeline.query('Give a brief about comparison of legacy braking system and brake by wire systems')
    print(response)