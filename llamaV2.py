import os
import glob
import json
import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

# ================================
# 🗂️ 1. CONFIGURAÇÃO DA BASE LOCAL
# ================================

PASTA_DOCUMENTOS = "C:/Users/Leonardo Borges/Desktop/ChatBotIA/Modelo3-LLAMA/Base/"  # pasta onde ficarão seus arquivos .pdf, .docx, .txt

if not os.path.exists(PASTA_DOCUMENTOS):
    os.makedirs(PASTA_DOCUMENTOS)
    print(f"📂 Pasta '{PASTA_DOCUMENTOS}' criada. Adicione seus arquivos nela e execute novamente.")
    exit()

documentos = []

for arquivo in os.listdir(PASTA_DOCUMENTOS):
    caminho = os.path.join(PASTA_DOCUMENTOS, arquivo)
    if arquivo.endswith(".pdf"):
        documentos.append(PyPDFLoader(caminho).load())
    elif arquivo.endswith(".txt"):
        documentos.append(TextLoader(caminho).load())
    elif arquivo.endswith(".docx"):
        documentos.append(Docx2txtLoader(caminho).load())

# Achata a lista (pois cada loader retorna uma lista)
documentos = [doc for sublist in documentos for doc in sublist]

if not documentos:
    print(f"⚠️ Nenhum documento encontrado na pasta '{PASTA_DOCUMENTOS}'.")
    exit()

print(f"✅ {len(documentos)} documentos carregados da pasta '{PASTA_DOCUMENTOS}'.")

# ==================================
# ✂️ 2. DIVISÃO EM PARTES (CHUNKS)
# ==================================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_divididos = splitter.split_documents(documentos)
print(f"📄 Total de {len(docs_divididos)} partes criadas.")

# ==================================
# 🧬 3. GERAÇÃO DE EMBEDDINGS
# ==================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ==================================
# 🧠 4. CRIAÇÃO DO BANCO VETORIAL
# ==================================
banco = FAISS.from_documents(docs_divididos, embeddings)

# ==================================
# 🗣️ 5. CONFIGURAÇÃO DO MODELO OLLAMA
# ==================================
modelo_ollama = "llama3"  # altere aqui para outro modelo (ex: phi3, mistral, gemma, neural-chat, etc.)
llm = OllamaLLM(model=modelo_ollama)

# ==================================
# 🔎 6. CRIAÇÃO DO SISTEMA DE PERGUNTAS
# ==================================
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=banco.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

print("\n🤖 Base de conhecimento pronta! Agora posso te ajudar com base nos arquivos carregados.\n")

# ==================================
# 💬 7. LOOP DE CHAT INTERATIVO
# ==================================
while True:
    pergunta = input("Você: ").strip()
    if pergunta.lower() in ["sair", "fim", "exit"]:
        print("👋 Até logo!")
        break

    resposta = qa(pergunta)
    print("\nAssistente:", resposta["result"])
    print("-" * 50)
