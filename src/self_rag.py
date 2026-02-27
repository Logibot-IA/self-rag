from rag_pipeline import get_llm

def self_rag(query, retriever):
    llm = get_llm()

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Contexto:
    {context}

    Pergunta:
    {query}

    Responda usando apenas o contexto.
    """

    response = llm.invoke(prompt).content

    # Auto-avaliação simples
    critique_prompt = f"""
    Pergunta: {query}
    Resposta: {response}

    A resposta está bem fundamentada no contexto?
    Responda apenas SIM ou NAO.
    """

    critique = llm.invoke(critique_prompt).content.strip()

    if "NAO" in critique.upper():
        refine_prompt = f"""
        Refaça a resposta com mais precisão e fundamentação no contexto:

        Contexto:
        {context}

        Pergunta:
        {query}
        """

        response = llm.invoke(refine_prompt).content

    return response, context