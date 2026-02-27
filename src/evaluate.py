from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from rag_pipeline import get_llm
from langchain_openai import OpenAIEmbeddings
import os

def run_evaluation(question, answer, context):
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[context]]
    }

    dataset = Dataset.from_dict(data)

    llm = get_llm()

    embeddings = OpenAIEmbeddings(
        base_url=os.getenv("DO_BASE_URL"),
        api_key=os.getenv("DO_API_KEY"),
        model="text-embedding-3-large"
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings
    )

    print(result)