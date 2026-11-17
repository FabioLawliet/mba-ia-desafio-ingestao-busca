
import os
import sys
import requests

from dotenv import load_dotenv
from providers import get_llm, get_embeddings, PROVIDER
from search import buscar_contexto

load_dotenv()

PROMPT = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO,
  responda exatamente:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.
- Se o contexto não apresentar informações suficientes,
  utilize a resposta padrão.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:

Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO".
"""

RESPOSTA_SEMINFO = ("Não tenho informações necessárias para responder sua pergunta.")

def responder(pergunta: str) -> str:
    contexto = buscar_contexto(pergunta)

    if not contexto.strip():
        return RESPOSTA_SEMINFO

    prompt = PROMPT.format(
        contexto=contexto,
        pergunta=pergunta
    )

    llm = get_llm()
    resposta = llm.invoke(prompt)
    return resposta.content.strip()


def main():

    print("=" * 60)
    print("CONSULTA AO DOCUMENTO PDF")
    print("=" * 60)

    while True:
        pergunta = input("Faça uma pergunta sobre o PDF (ou digite sair): ")

        if pergunta.lower().strip() == "sair":
            break

        if not pergunta.strip():
            continue

        try:
            print("\nConsultando documento...")
            resposta = responder(pergunta)
            print(f"\nRESPOSTA: {resposta}\n")

        except Exception as erro:
            print(f"\nErro ao processar a pergunta: {erro}")
            break


if __name__ == "__main__":
    main()
