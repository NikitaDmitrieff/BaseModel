SYSTEM_PROMPT_TEMPLATE = """
Vous êtes un coach d'orientation expérimenté, spécialisé dans l'accompagnement des étudiants. Soyez informatif, serviable et encourageant.
"""

USER_PROMPT_TEMPLATE = """Répondez à la question suivante: {question}
Réponse:"""

RAG_SYSTEM_PROMPT_TEMPLATE = """
Vous êtes un coach d'orientation expérimenté, spécialisé dans l'accompagnement des étudiants. Soyez informatif, serviable et encourageant.
"""

RAG_USER_PROMPT_TEMPLATE = """Répondez à la question encadrée par %%% en basant votre réponse sur le contexte fourni entre ###:

%%% QUESTION

{question}

%%%

###

{context}

###

Réponse:"""

HYDE_PROMPT_TEMPLATE = """Ecris un passage qui serait susceptible de répondre à la question suivante.

Question: {QUESTION}

Passage:"""


def rag_prompt_format(
    user_question: str, context: str, system_template: str, user_template: str
) -> tuple[str, str]:

    system_prompt = system_template
    user_prompt = user_template.format(question=user_question, context=context)

    return system_prompt, user_prompt


def prompt_format(
    user_question: str, system_template: str, user_template: str
) -> tuple[str, str]:

    system_prompt = system_template
    user_prompt = user_template.format(question=user_question)

    return system_prompt, user_prompt
