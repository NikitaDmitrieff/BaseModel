SYSTEM_PROMPT_TEMPLATE = """
You are an experienced guidance counselor, specializing in supporting students. Be informative, helpful, and encouraging.
"""

USER_PROMPT_TEMPLATE = """Answer the following question: {question}
Answer:"""

RAG_SYSTEM_PROMPT_TEMPLATE = """
You are an experienced guidance counselor, specializing in supporting students. Be informative, helpful, and encouraging.
"""

RAG_USER_PROMPT_TEMPLATE = """Answer the question framed by %%% based on the context provided between ###:

%%% QUESTION

{question}

%%%

###

{context}

###

Answer:"""

HYDE_PROMPT_TEMPLATE = """Write a passage that would likely answer the following question.

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
