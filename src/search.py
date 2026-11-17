import os

from dotenv import load_dotenv
from langchain_postgres import PGVector
from providers import get_embeddings

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

def buscar_contexto(pergunta: str) -> str:

    embeddings = get_embeddings()

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True
    )

    # Buscar os 10 resultados mais relevantes
    resultados = vector_store.similarity_search_with_score(
        pergunta,
        k=10
    )

    # Concatenar os textos encontrados
    contextos = []

    for documento, score in resultados:
        pagina = documento.metadata.get("page", "N/A")

        contextos.append(
            f"Página: {pagina + 1 if isinstance(pagina, int) else pagina}\n"
            f"Conteúdo: {documento.page_content}"
        )

    return "\n\n".join(contextos)

