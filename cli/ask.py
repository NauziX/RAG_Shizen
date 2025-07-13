"""Pregunta al chatbot desde línea de comandos."""
from __future__ import annotations

import argparse

from src import ask


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el RAG de Shizen")
    parser.add_argument("question", nargs="+", help="Texto de la pregunta")
    args = parser.parse_args()

    query = " ".join(args.question)
    print("»", ask(query))


if __name__ == "__main__":
    main()
