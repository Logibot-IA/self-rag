import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset


load_dotenv()

BASE_URL = os.getenv("DO_BASE_URL")
API_KEY = os.getenv("DO_API_KEY")
MODEL = os.getenv("DO_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")


def build_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings
    )

    return vectordb, embeddings


def self_rag(query, retriever, llm):
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

    # Auto-crítica simples
    critique_prompt = f"""
    Pergunta: {query}
    Resposta: {response}

    A resposta está fundamentada no contexto?
    Responda apenas SIM ou NAO.
    """

    critique = llm.invoke(critique_prompt).content

    if "NAO" in critique.upper():
        refine_prompt = f"""
        Refaça a resposta usando melhor o contexto.

        Contexto:
        {context}

        Pergunta:
        {query}
        """

        response = llm.invoke(refine_prompt).content

    return response, context

def generate_ground_truth(question, context, llm):
    prompt = f"""
    Você é um especialista. 
    Responda a pergunta usando APENAS o contexto abaixo.
    Seja preciso e objetivo.

    Pergunta:
    {question}

    Contexto:
    {context}

    Resposta:
    """

    response = llm.invoke(prompt)
    return response.content

def run_ragas(question, answer, context, ground_truth, llm, embeddings):
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[context]],
        "ground_truth": [ground_truth]
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision
        ],
        llm=llm,
        embeddings=embeddings
    )

    print("\n\n\nResultado RAGAS:\n")
    print(result)


def main():
    print("\nIndexando PDF...\n")
    vectordb, embeddings = build_vectorstore("Apostila_Logica.pdf")

    retriever = vectordb.as_retriever()

    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        temperature=0
    )

    query = input("\n\n\nPergunta: ")

    answer, context = self_rag(query, retriever, llm)

    print("\n\n\nResposta:\n")
    print(answer)

    ground_truth = generate_ground_truth(query, context, llm)

    run_ragas(query, answer, context, ground_truth, llm, embeddings)


if __name__ == "__main__":
    main()