"""Mecanismo de troca de provedor de IA.

Controlado pela variável de ambiente AI_PROVIDER no .env:

    AI_PROVIDER=google   -> Google AI Studio (API remota)
    AI_PROVIDER=ollama   -> Ollama (local, sem cota de uso)
"""

import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("AI_PROVIDER", "google").strip().lower()


def _validar_provider():
    if PROVIDER not in ("google", "ollama"):
        raise ValueError(
            f"AI_PROVIDER inválido: {PROVIDER!r}. Use 'google' ou 'ollama' no .env"
        )


def get_embeddings():
    _validar_provider()

    if PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL")
    )


def get_llm():
    """Retorna o LLM de chat escolhido em AI_PROVIDER."""
    _validar_provider()

    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_CHAT_MODEL", "gemma3:4b"),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_CHAT_MODEL")
    )
