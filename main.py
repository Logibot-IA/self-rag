import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

load_dotenv()

BASE_URL = os.getenv("DO_BASE_URL")
API_KEY = os.getenv("DO_API_KEY")
MODEL = os.getenv("DO_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")

test_queries = [
    "O que é lógica proposicional segundo a apostila?",
    "Como a apostila define uma proposição?",
    "O que são conectivos lógicos e quais são apresentados no material?",
    "O que é uma tabela-verdade e para que ela é utilizada?",
    "Como a apostila define tautologia, contradição e contingência?"
]

ground_truths = [
    "Lógica proposicional é o ramo da lógica que estuda proposições e as relações entre elas por meio de conectivos lógicos.",
    "Proposição é toda sentença declarativa que pode ser classificada como verdadeira ou falsa, mas não ambas.",
    "Conectivos lógicos são operadores que conectam proposições, como negação (¬), conjunção (∧), disjunção (∨), condicional (→) e bicondicional (↔).",
    "Tabela-verdade é um método utilizado para determinar o valor lógico de proposições compostas a partir dos valores lógicos das proposições simples.",
    "Tautologia é uma proposição composta que é sempre verdadeira; contradição é sempre falsa; contingência é aquela que pode ser verdadeira ou falsa dependendo dos valores das proposições componentes."
]


def build_vectorstore():
    loader = DirectoryLoader("../docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(chunks, embedding=embeddings)
    return vectordb, embeddings


def self_rag(query, retriever, llm):
    docs = retriever.get_relevant_documents(query)
    contexts = [d.page_content for d in docs]
    context = "\n\n".join(contexts)

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

    return response, contexts


def run_ragas(ragas_data, llm, embeddings):
    dataset = Dataset.from_list(ragas_data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings
    )

    print("\n=== RESULTADOS RAGAS ===")
    print(result)

    df = result.to_pandas()
    print("\nDetalhes por query:")
    print(df.to_string())

    return result


def main():
    print("\nIndexando PDFs de ../docs/ ...\n")
    vectordb, embeddings = build_vectorstore()
    retriever = vectordb.as_retriever()

    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        temperature=0
    )

    print("Coletando respostas para avaliacao RAGAS...\n")
    ragas_data = []
    for i, query in enumerate(test_queries):
        print(f"  [{i+1}/{len(test_queries)}] {query}")
        answer, contexts = self_rag(query, retriever, llm)
        ragas_data.append({
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truths[i]
        })

    run_ragas(ragas_data, llm, embeddings)


if __name__ == "__main__":
    main()
