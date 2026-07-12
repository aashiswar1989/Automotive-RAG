from pathlib import Path

PROJECT_NAME = "Automotive_RAG"

file_list = [
    f'src/{PROJECT_NAME}/__init__.py',
    f'src/{PROJECT_NAME}/config.py',
    f'src/{PROJECT_NAME}/ingestion/__init__.py',
    f'src/{PROJECT_NAME}/embeddings/__init__.py',
    f'src/{PROJECT_NAME}/retrieval/__init__.py',
    f'src/{PROJECT_NAME}/chains/__init__.py',
    f'src/{PROJECT_NAME}/main.py',
    'data/raw/',
    'data/processed/',
    'data/vectorstore/',
    'utils/__init__.py',
    'utils/logger.py',
    'requirements.txt',
    'notebooks/experiments.ipynb',
    'configs/settings.yaml'
]

for file_path in file_list:
    if not file_path.endswith('/'):
        file, parent = Path(file_path), Path(file_path).parent

        if not parent.is_dir():
            parent.mkdir(parents=True, exist_ok=True)

        if not file.exists():
            file.touch(exist_ok=True)
    else:
        parent = Path(file_path)
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)