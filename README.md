README – Sistema de Perguntas e Respostas com LLama + FAISS + LangChain

Este projeto implementa um chat inteligente baseado em documentos locais, utilizando:

LangChain

FAISS (vetor store)

Embeddings da HuggingFace (MiniLM)

Modelo LLama rodando via Ollama

PDFs, TXT e DOCX como base de conhecimento

Você adiciona os arquivos em uma pasta, o sistema cria embeddings, monta um banco vetorial e responde perguntas baseadas no conteúdo.

🧩 1. Requisitos do Sistema
✔ Python

Python 3.9+

✔ Dependências externas obrigatórias

Ollama instalado
Download: https://ollama.com/download

Após instalar, rode:

ollama run llama3


ou outro modelo (phi3, gemma, mistral, etc.)

✔ GPU (opcional, mas recomendado)

NVIDIA / AMD ou Apple Silicon

Ollama acelera automaticamente se GPU estiver disponível

📁 2. Estrutura de Arquivos do Projeto
/seu-projeto
│
├── llamaV2.py                      → script principal
├── Base/                           → pasta com seus documentos
│   ├── arquivo1.pdf
│   ├── arquivo2.docx
│   ├── arquivo3.txt
│   └── ...
└── README.md

📂 3. Criar a Pasta Base de Conhecimento

O script usa esta pasta:

C:/Users/Leonardo Borges/Desktop/ChatBotIA/Modelo3-LLAMA/Base/


Mas você pode mudar o caminho alterando:

PASTA_DOCUMENTOS = "C:/.../Base/"


Dentro dela, coloque arquivos:

.pdf

.docx

.txt

Se a pasta estiver vazia, o script avisa e encerra.

📦 4. Instalar Dependências Python

Instale os pacotes necessários:

pip install langchain langchain-community langchain-ollama faiss-cpu sentence-transformers pypdf python-docx


Se quiser usar FAISS GPU:

pip install faiss-gpu

🔧 5. O que o Script Faz

O arquivo llamaV2.py executa todo o pipeline de RAG (Retrieval Augmented Generation):

📌 1. Carrega os documentos

Usando loaders:

PDF → PyPDFLoader

DOCX → Docx2txtLoader

TXT → TextLoader

📌 2. Divide em chunks

Trechos de:

chunk_size=1000

chunk_overlap=100

📌 3. Gera embeddings

Usando:

sentence-transformers/all-MiniLM-L6-v2


Rápido e eficiente.

📌 4. Cria o banco vetorial FAISS

Armazena embeddings localmente para buscas rápidas.

📌 5. Usa o modelo LLama no Ollama
modelo_ollama = "llama3"
llm = OllamaLLM(model=modelo_ollama)


Você pode trocar por:

mistral

phi3

neural-chat

gemma

llama3:instruct

etc.

📌 6. Cria o sistema de QA com LangChain

O sistema faz:

Busca semântica no FAISS

Seleciona trechos relevantes

Envia ao modelo LLama gerar a resposta

Retorna resposta + fontes

📌 7. Inicia o chat interativo

Comandos para sair:

sair

fim

exit

🚀 6. Como Executar o Script

No terminal:

python llamaV2.py


Se estiver tudo certo, verá:

📄 Total de X partes criadas.
🤖 Base de conhecimento pronta!


E poderá perguntar:

Você: O que diz o contrato sobre rescisão?

💡 7. Personalizar o Modelo

No código:

modelo_ollama = "llama3"


Substitua por qualquer modelo instalado no Ollama:

ollama pull phi3
ollama pull mistral
ollama pull gemma
