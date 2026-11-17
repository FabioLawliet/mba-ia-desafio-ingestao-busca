import os

from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from providers import get_embeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "document.pdf"

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")


def main():
    print("Carregando PDF para o banco de dados.")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF não encontrado: {PDF_PATH}")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Quantidade de chunks: {len(chunks)}")

    embeddings = get_embeddings()
    textos = [chunk.page_content for chunk in chunks]
    vetores = embeddings.embed_documents(textos)


    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=True
    )

    vector_store.add_embeddings(
        texts=textos,
        embeddings=vetores,
        metadatas=[chunk.metadata for chunk in chunks]
    )

    print("PDF carregado para o Banco de dados com sucesso!")


if __name__ == "__main__":
    main()
