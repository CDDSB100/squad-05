"""Geração de respostas multi-turno com LLM."""

import ollama

from core.config import settings
from core.schemas import UserProfile
from profiling.profiler import build_system_prompt


def generate(
    question: str,
    context: str,
    history: list[dict[str, str]],
    profile: UserProfile,
) -> str:
    """Gera resposta do LLM considerando contexto RAG, histórico e perfil do usuário.

    Args:
        question: Pergunta atual do usuário.
        context: Texto de contexto recuperado via RAG (chunks concatenados).
        history: Lista de mensagens anteriores no formato [{"role": "user"|"assistant", "content": "..."}].
        profile: Perfil do usuário com nome e nível de expertise.

    Returns:
        Texto da resposta gerada pelo LLM.
    """
    messages: list[dict[str, str]] = []

    # System prompt baseado no perfil
    messages.append({"role": "system", "content": build_system_prompt(profile)})

    # Histórico de conversa
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Pergunta atual com contexto RAG injetado
    user_content = (
        f"Contexto:\n{context}\n\nPergunta: {question}" if context else f"Pergunta: {question}"
    )
    messages.append({"role": "user", "content": user_content})

    response = ollama.chat(
        model=settings.chat_model,
        messages=messages,
        options={"num_predict": settings.llm_max_tokens},
    )

    return str(response["message"]["content"])
