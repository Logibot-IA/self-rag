# Self-RAG

Sistema de Retrieval-Augmented Generation com auto-reflexão e avaliação por métricas RAGAS.

## Descrição

Implementação de Self-RAG que processa documentos PDF, realiza busca semântica e gera respostas com mecanismo de auto-crítica. O sistema avalia a qualidade das respostas usando métricas de fidelidade, relevância e precisão de contexto.

## Funcionalidades

- Indexação de documentos PDF com ChromaDB
- Embeddings via OpenAI (`text-embedding-3-large` por padrão)
- Self-RAG com auto-crítica de respostas
- Avaliação automática com RAGAS
- Geração de respostas via OpenAI (`gpt-5.5` por padrão)

## Requisitos

- Python 3.8+
- Chave de API da OpenAI

## Instalação

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

## Configuração

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sk-sua_chave_openai
OPENAI_MODEL=gpt-5.5
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_REASONING_EFFORT=medium
CHROMA_PERSIST_DIR=./chroma_self_db_openai
CHROMA_COLLECTION_NAME=self_rag_contexts_openai
```

## Uso

Coloque os PDFs na pasta `docs/` e execute:

```bash
python main.py
```

O sistema:
1. Indexa o PDF em chunks de 800 caracteres
2. Executa as perguntas de benchmark configuradas em `main.py`
3. Busca contexto relevante no vector store
4. Gera resposta com auto-crítica
5. Salva métricas RAGAS em CSV

## Métricas RAGAS

- **Faithfulness**: Fidelidade da resposta ao contexto
- **Answer Relevancy**: Relevância da resposta à pergunta
- **Context Precision**: Precisão do contexto recuperado

## Estrutura do Projeto

```
self-rag/
├── main.py              # Código principal
├── requirements.txt     # Dependências
├── .env                 # Variáveis de ambiente
└── docs/                # PDFs usados como base de conhecimento
```
