# 📚 Self-RAG com Avaliação Automática usando RAGAS

Implementação de um sistema **Self-RAG dockerizado**, utilizando:

-   Ollama (LLM local)
-   Chroma (Banco Vetorial)
-   RAGAS (Avaliação automática)
-   FastAPI
-   Docker

------------------------------------------------------------------------

# 🎯 Objetivo do Projeto

Este projeto implementa:

-   ✅ Retrieval-Augmented Generation (RAG)
-   ✅ Self-Refinement (Self-RAG)
-   ✅ Avaliação automática com métricas acadêmicas
-   ✅ Ambiente 100% dockerizado
-   ✅ Preparado para ingestão futura de PDF (livro de lógica de
    programação)

------------------------------------------------------------------------

# 🧠 Conceito de Self-RAG

Fluxo do sistema:

Pergunta → Recuperação → Resposta inicial → Autoavaliação → Refinamento

------------------------------------------------------------------------

# 🏗 Arquitetura

Usuário\
↓\
API (FastAPI)\
↓\
Chroma (Vetores)\
↓\
Ollama (LLM local)\
↓\
Resposta final

Serviços Docker:

-   ollama
-   chroma
-   api
-   evaluator

------------------------------------------------------------------------

# 📁 Estrutura do Projeto

    self-rag/
    │
    ├── docker-compose.yml
    ├── Dockerfile
    ├── requirements.txt
    ├── README.md
    │
    └── app/
        ├── main.py
        ├── rag.py
        ├── ingest.py
        ├── evaluate.py
        ├── dataset.json
        └── data/
            └── livro.pdf

------------------------------------------------------------------------

# 🚀 Como Executar

## 1️⃣ Subir os containers

``` bash
docker compose up --build
```

## 2️⃣ Baixar modelo no Ollama

``` bash
docker exec -it ollama ollama pull mistral
```

## 3️⃣ Ingerir PDF

Coloque o livro em:

app/data/livro.pdf

Execute:

``` bash
docker exec -it self-rag-api python -c "from app.ingest import ingest_pdf; ingest_pdf('app/data/livro.pdf')"
```

## 4️⃣ Fazer perguntas

Abra:

http://localhost:8000/ask?question=O que é algoritmo?

## 5️⃣ Rodar avaliação com RAGAS

``` bash
docker compose run evaluator
```

------------------------------------------------------------------------

# 📊 Métricas Avaliadas

-   Faithfulness
-   Answer Relevancy
-   Context Precision
-   Context Recall

------------------------------------------------------------------------

# 🔥 Possíveis Evoluções

-   Re-ranking
-   Relatório automático em PDF
-   Interface web
-   CI/CD para avaliação contínua
-   Comparação entre modelos

------------------------------------------------------------------------

# 👨‍💻 Autor

Martiniano Gomes Barros Cirqueira Neto

------------------------------------------------------------------------

Licença: MIT
