from fastapi import FastAPI
from app.rag import self_rag

app = FastAPI()

@app.get("/ask")
def ask(question: str):
    response = self_rag(question)
    return {"answer": response}