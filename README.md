# Self-RAG

Sistema de Retrieval-Augmented Generation com auto-reflexão e avaliação por métricas RAGAS.

## Descrição

Implementação de Self-RAG que processa documentos PDF, realiza busca semântica e gera respostas com mecanismo de auto-crítica. O sistema avalia a qualidade das respostas usando métricas de fidelidade, relevância e precisão de contexto.

## Funcionalidades

- Indexação de documentos PDF com ChromaDB
- Embeddings via HuggingFace (all-MiniLM-L6-v2)
- Self-RAG com auto-crítica de respostas
- Avaliação automática com RAGAS
- Suporte a LLMs compatíveis com OpenAI API

## Requisitos

- Python 3.8+
- Chave de API para LLM (DigitalOcean AI ou OpenAI compatível)
- Token HuggingFace (opcional)

## Instalação

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

## Configuração

Crie um arquivo `.env` na raiz do projeto:

```env
DO_API_KEY=sua_chave_api
DO_BASE_URL=https://inference.do-ai.run/v1
DO_MODEL=openai-gpt-oss-120b
HF_TOKEN=seu_token_huggingface
```

## Uso

Coloque seu documento PDF na raiz do projeto e execute:

```bash
python main.py
```

O sistema:
1. Indexa o PDF em chunks de 800 caracteres
2. Aguarda sua pergunta via input
3. Busca contexto relevante no vector store
4. Gera resposta com auto-crítica
5. Exibe métricas RAGAS de avaliação

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
└── Apostila_Logica.pdf  # Documento de exemplo
```