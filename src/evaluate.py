import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_ollama import ChatOllama
from src.rag import self_rag

DATASET_PATH = "src/dataset.json"

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = []
answers = []
ground_truths = []
contexts = []

for item in data:

    q = item["question"]
    gt = item["ground_truth"]

    response, docs = self_rag(q)

    questions.append(q)
    answers.append(response)
    ground_truths.append(gt)

    # Context dummy — pode evoluir depois para capturar real retrieval
    contexts.append([doc.page_content for doc in docs])

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
})

llm = ChatOllama(
    model="mistral",
    base_url="http://ollama:11434"
)

result = evaluate(
    dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ],
    llm=llm,
)

print(result)