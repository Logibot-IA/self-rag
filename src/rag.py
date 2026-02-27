import requests
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/chat"

PERSIST_DIR = "/app/chroma_db"

MODEL_GENERATE_NAME = "phi3:mini"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)


# def generate(prompt: str):

#     response = requests.post(
#         OLLAMA_GENERATE_URL,
#         json={
#             "model": "mistral",
#             "prompt": prompt,
#             "stream": False
#         }
#     )

#     return response.json()["response"]

# def generate(prompt):
#     response = requests.post(
#         "http://ollama:11434/api/chat",
#         json={
#             "model": MODEL_GENERATE_NAME,
#             "messages": [
#                 {"role": "user", "content": prompt}
#             ],
#             "stream": False
#         }
#     )

#     print("STATUS:", response.status_code)
#     print("JSON:", response.text)

#     return response.json()

def generate(prompt):
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": MODEL_GENERATE_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
    )

    if response.status_code != 200:
        print("Erro HTTP:", response.text)
        return "Erro na geração."

    data = response.json()

    if "message" in data:
        return data["message"]["content"]

    if "response" in data:
        return data["response"]

    if "error" in data:
        print("Erro Ollama:", data["error"])
        return "Erro na geração."

    print("Formato inesperado:", data)
    return "Erro inesperado."


def self_rag(question: str):

    docs = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    initial_prompt = f"""
Use o contexto abaixo para responder:

{context}

Pergunta:
{question}
"""

    answer = generate(initial_prompt)

    critique_prompt = f"""
Pergunta: {question}

Resposta: {answer}

A resposta está completa e fundamentada no contexto?
Responda apenas SIM ou NÃO.
"""

    critique = generate(critique_prompt)

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