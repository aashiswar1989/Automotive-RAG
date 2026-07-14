import logging
from pathlib import Path

LOG_DIR = Path('logs')
LOG_FILE = "file_organizer.log"

LOG_PATH = LOG_DIR/LOG_FILE

def get_logger(name:str)->logging.Logger:

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    if not LOG_DIR.is_dir():
        LOG_DIR.mkdir(exist_ok=True)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(lineno)d | %(levelname)s | %(name)s - %(message)s"
    )

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Logger has been configured successfully.")
    return logger