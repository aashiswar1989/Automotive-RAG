# ollama_utils.py
import subprocess
import time
import httpx

OLLAMA_HOST = "http://localhost:11434"


def is_ollama_running(host: str = OLLAMA_HOST) -> bool:
    try:
        httpx.get(host, timeout=1)
        return True
    except httpx.ConnectError:
        return False


def ensure_ollama_running(host: str = OLLAMA_HOST, timeout_s: int = 30) -> None:
    if is_ollama_running(host):
        return  # already running, skip

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(timeout_s):
        if is_ollama_running(host):
            return
        time.sleep(1)

    raise RuntimeError(f"Ollama did not become reachable within {timeout_s}s")