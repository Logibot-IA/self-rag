import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain.chat_models import ChatOllama
from app.rag import self_rag

# Carregar dataset
with open("app/dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = []
answers = []
contexts = []
ground_truths = []

for item in data:
    q = item["question"]
    gt = item["ground_truth"]

    response = self_rag(q)

    questions.append(q)
    answers.append(response)
    contexts.append(["contexto recuperado"])
    ground_truths.append(gt)

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
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
)

print(result)