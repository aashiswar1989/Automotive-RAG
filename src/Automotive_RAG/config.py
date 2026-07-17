from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class DataConfig:
    source_pdf : str
    source_md : str
    vector_db : str

@dataclass
class ChunkingConfig:
    chunk_size : int
    chunk_overlap : int

@dataclass
class EmbeddingConfig:
    model: str
    provider: str
    collection_name: str
    persist_dir: str

@dataclass
class LLMConfig:
    model : str
    provider: str
    temperature: float

@dataclass
class RetrieverConfig:
    k : int
    search_type : str


@dataclass
class ConfigManager:
    data: DataConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    retriever: RetrieverConfig


def ConfigLoader(config_path: str) -> ConfigManager:
    with open(config_path, 'r') as f:
        config_info = f.read()
    config = yaml.safe_load(config_info)

    data = DataConfig(**config['data'])
    chunking = ChunkingConfig(**config['chunking'])
    embedding = EmbeddingConfig(**config['embedding'])
    llm = LLMConfig(**config['llm'])
    retriever = RetrieverConfig(**config['retriever'])

    return ConfigManager(
        data=data,
        chunking=chunking,
        embedding=embedding,
        llm=llm,
        retriever=retriever,
        )
