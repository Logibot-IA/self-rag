import requests
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

OLLAMA_URL = "http://ollama:11434/api/generate"

embeddings = OllamaEmbeddings(
    model="mistral",
    base_url="http://ollama:11434"
)

vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

def generate(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def self_rag(question: str):

    # 🔎 Recuperação
    docs = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # 🧠 Resposta inicial
    initial_prompt = f"""
    Use o contexto abaixo para responder:

    {context}

    Pergunta:
    {question}
    """

    answer = generate(initial_prompt)

    # 🔍 Auto-avaliação
    critique_prompt = f"""
    Pergunta: {question}

    Resposta: {answer}

    A resposta está completa e fundamentada no contexto?
    Responda apenas SIM ou NÃO.
    """

    critique = generate(critique_prompt)

    # 🔁 Refinamento
    if "NÃO" in critique.upper():
        refine_prompt = f"""
        Melhore a resposta abaixo usando melhor o contexto:

        Contexto:
        {context}

        Resposta anterior:
        {answer}
        """

        answer = generate(refine_prompt)

    return answer