# llamaV2_fix.py
import os
import glob
import sys
import json

# ---------- Imports resilientes (tenta múltiplos caminhos) ----------
print("🔎 Tentando importar dependências do LangChain/FAISS/Ollama...")

# Document loaders
PyPDFLoader = None
TextLoader = None
Docx2txtLoader = None
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    print("✔️ Usando langchain_community.document_loaders")
except Exception:
    try:
        from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
        print("✔️ Usando langchain.document_loaders")
    except Exception:
        print("❌ Não encontrou loaders document_loaders automaticamente. Instale langchain-community.")
        # Não aborta aqui; o erro aparecerá se tentar usar loaders.

# Text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("✔️ Usando langchain_text_splitters.RecursiveCharacterTextSplitter")
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✔️ Usando langchain.text_splitter.RecursiveCharacterTextSplitter")
    except Exception:
        RecursiveCharacterTextSplitter = None
        print("❌ RecursiveCharacterTextSplitter não encontrado - instale langchain-text-splitters")

# Vectorstore FAISS
FAISS = None
try:
    from langchain_community.vectorstores import FAISS
    print("✔️ Usando langchain_community.vectorstores.FAISS")
except Exception:
    try:
        from langchain.vectorstores import FAISS
        print("✔️ Usando langchain.vectorstores.FAISS")
    except Exception:
        FAISS = None
        print("❌ FAISS import não encontrado - instale langchain-community e faiss-cpu")

# Embeddings (prefer sentence-transformers)
HuggingFaceEmbeddings = None
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    print("✔️ Usando langchain.embeddings.HuggingFaceEmbeddings")
except Exception:
    try:
        from langchain_hub import HuggingFaceEmbeddings
        print("✔️ Usando langchain_hub.HuggingFaceEmbeddings (fallback)")
    except Exception:
        HuggingFaceEmbeddings = None
        print("❌ HuggingFaceEmbeddings não encontrado - instale sentence-transformers e langchain embeddings")

# Ollama LLM
OllamaLLM = None
try:
    from langchain_ollama import OllamaLLM
    print("✔️ Usando langchain_ollama.OllamaLLM")
except Exception:
    try:
        # fallback: older names
        from langchain_ollama import Ollama as OllamaLLM
        print("✔️ Usando langchain_ollama.Ollama (alias)")
    except Exception:
        OllamaLLM = None
        print("❌ langchain_ollama não disponível - instale langchain-ollama or use HF model")

# Retrieval chain (compatibilidade)
RetrievalQA = None
create_retrieval_chain = None
try:
    # primeira tentatica: import direto (algumas versões)
    from langchain.chains import RetrievalQA
    print("✔️ Usando langchain.chains.RetrievalQA")
    RetrievalQA = RetrievalQA
except Exception:
    try:
        from langchain.chains.retrieval_qa.base import RetrievalQA
        print("✔️ Usando langchain.chains.retrieval_qa.base.RetrievalQA")
        RetrievalQA = RetrievalQA
    except Exception:
        try:
            from langchain.chains import create_retrieval_chain
            print("✔️ Usando langchain.chains.create_retrieval_chain")
            create_retrieval_chain = create_retrieval_chain
        except Exception:
            print("❌ RetrievalQA / create_retrieval_chain não encontrados - instale langchain atualizado")

# ---------- Verificações mínimas ----------
if RecursiveCharacterTextSplitter is None or FAISS is None or HuggingFaceEmbeddings is None:
    print("\n❗ Aviso: Seu ambiente está faltando algumas bibliotecas essenciais.")
    print("Sugestão: pip install -U langchain langchain-core langchain-community langchain-text-splitters langchain-ollama faiss-cpu pypdf python-docx sentence-transformers\n")
    # Não abortamos; o script tentará rodar e mostrará erros mais claros.

# ---------- Configurações ----------
PASTA_DOCUMENTOS = r"C:\Users\Leonardo Borges\Desktop\ChatBotIA\Modelo3-LLAMA\Base"  # <- ajuste aqui
if not os.path.exists(PASTA_DOCUMENTOS):
    os.makedirs(PASTA_DOCUMENTOS)
    print(f"📁 Pasta criada em: {PASTA_DOCUMENTOS}. Coloque seus .pdf/.docx/.txt lá e reexecute.")
    sys.exit(0)

# ---------- Carregar documentos ----------
def carregar_docs(pasta):
    docs = []
    patterns = ["*.pdf", "*.docx", "*.txt"]
    for p in patterns:
        for caminho in glob.glob(os.path.join(pasta, p)):
            print("📄 Lendo:", caminho)
            try:
                if caminho.lower().endswith(".pdf") and 'PyPDFLoader' in globals() and PyPDFLoader is not None:
                    docs.extend(PyPDFLoader(caminho).load())
                elif caminho.lower().endswith(".docx") and 'Docx2txtLoader' in globals() and Docx2txtLoader is not None:
                    docs.extend(Docx2txtLoader(caminho).load())
                elif caminho.lower().endswith(".txt") and 'TextLoader' in globals() and TextLoader is not None:
                    docs.extend(TextLoader(caminho).load())
                else:
                    # Fallback: tenta leitura simples
                    with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        docs.append({"page_content": text, "metadata": {"source": caminho}})
            except Exception as e:
                print("⚠️ Erro ao ler", caminho, ":", e)
    return docs

documentos = carregar_docs(PASTA_DOCUMENTOS)
if not documentos:
    print("⚠️ Nenhum documento carregado. Verifique a pasta e os loaders.")
    sys.exit(1)

print(f"✅ {len(documentos)} blocos/documentos carregados (antes do split).")

# ---------- Split em chunks ----------
if RecursiveCharacterTextSplitter is None:
    print("❗ Sem text splitter instalado, não será feito split. Siga a sugestão de instalação.")
    docs_chunks = documentos
else:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    # se documentos já estiverem no formato de Document do langchain, splitter espera esse formato
    # tratamos caso sejam dicts: transformamos em objeto simples
    from types import SimpleNamespace
    docs_for_split = []
    for d in documentos:
        if isinstance(d, dict):
            docs_for_split.append(SimpleNamespace(page_content=d.get("page_content",""), metadata=d.get("metadata",{})))
        else:
            docs_for_split.append(d)
    docs_chunks = splitter.split_documents(docs_for_split)
    print(f"📄 Após split: {len(docs_chunks)} partes.")

# ---------- Embeddings e FAISS ----------
if HuggingFaceEmbeddings is None or FAISS is None:
    print("❗ Não é possível criar FAISS/Embeddings: libs faltando.")
    sys.exit(1)

print("🧬 Criando embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("💾 Indexando no FAISS...")
banco = FAISS.from_documents(docs_chunks, emb)
print("✅ Index criado.")

# ---------- Configurar LLM Ollama (ou fallback para erro claro) ----------
if OllamaLLM is None:
    print("❗ OllamaLLM não encontrado. Verifique instalação langchain-ollama. Tentando continuar (mas não haverá LLM).")
    sys.exit(1)

modelo_ollama = "phi3"  # altere aqui para 'phi3' ou 'llama3', etc.
print(f"🔁 Inicializando OllamaLLM (modelo={modelo_ollama})...")
llm = OllamaLLM(model=modelo_ollama)

# ---------- Criar a cadeia de retrieval (compatível com várias versões) ----------
if RetrievalQA is not None:
    try:
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=banco.as_retriever(search_kwargs={"k":3}), return_source_documents=True)
        print("✔️ RetrievalQA criado via from_chain_type.")
    except Exception as e:
        print("⚠️ Falha ao criar RetrievalQA via from_chain_type:", e)
        qa = None
elif create_retrieval_chain is not None:
    try:
        # fallback usando create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        combine = create_stuff_documents_chain(llm)
        qa = create_retrieval_chain(banco.as_retriever(search_kwargs={"k":3}), combine)
        print("✔️ Retrieval chain criado via create_retrieval_chain.")
    except Exception as e:
        print("❌ Falha ao criar chain via create_retrieval_chain:", e)
        qa = None
else:
    print("❌ Não foi possível criar o chain de recuperação. Verifique langchain.")
    qa = None

if qa is None:
    print("❗ Erro: cadeia de QA não inicializada. Saindo.")
    sys.exit(1)

# ---------- Loop de interação ----------
print("\n🤖 Base pronta. Pergunte algo (digite 'fim' para sair):\n")
while True:
    q = input("Você: ").strip()
    if q.lower() in ("fim","sair","exit"):
        print("👋 Ok, tchau.")
        break
    print("🔎 Buscando trecho relevante...")
    try:
        resposta = qa({"query": q})
    except Exception as e:
        print("❌ Erro ao executar qa:", e)
        continue

    # resposta pode ser dict com 'result' ou 'answer' dependendo da versão
    answer = resposta.get("result") or resposta.get("answer") or resposta
    print("\nAssistente:", answer)
    print("-"*50)
    sources = resposta.get("source_documents") or resposta.get("source_documents", [])
    if sources:
        for i, s in enumerate(sources, 1):
            src = getattr(s, "metadata", None)
            if isinstance(src, dict):
                srcname = src.get("source", "Desconhecida")
            else:
                srcname = getattr(s, "metadata", {}).get("source", "Desconhecida")
            print(f"Fonte {i}: {srcname}")
    print("-"*50)