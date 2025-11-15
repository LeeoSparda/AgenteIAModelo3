import os
import sys
import subprocess
import time
import json
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ============================================================
# 1️⃣ Instala automaticamente pacotes necessários
# ============================================================
def instalar(pacote):
    try:
        __import__(pacote)
    except ImportError:
        print(f"📦 Instalando {pacote} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pacote])

for pkg in ["torch", "transformers", "accelerate", "sentencepiece", "requests"]:
    instalar(pkg)

# ============================================================
# 2️⃣ Configuração do Token Hugging Face
# ============================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

if not HF_TOKEN:
    print("\n🔑 Nenhum token Hugging Face detectado.")
    print("➡️  Se quiser usar Hugging Face, gere um em: https://huggingface.co/settings/tokens")
    HF_TOKEN = input("Cole seu token (ou pressione Enter para ignorar): ").strip()
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN

# ============================================================
# 3️⃣ Função de Chat com Hugging Face
# ============================================================
def chat_huggingface(model_name="microsoft/Phi-3-mini-4k-instruct"):
    print(f"\n🤖 Carregando modelo Hugging Face: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

    print("\n💬 Chat Hugging Face iniciado! Digite 'fim' para encerrar.\n")
    while True:
        msg = input("Você: ")
        if msg.lower() == "fim":
            break
        resposta = pipe(msg)[0]["generated_text"]
        print("Assistente:", resposta)

# ============================================================
# 4️⃣ Função de Chat com Ollama
# ============================================================
def chat_ollama(model="phi3"):
    print(f"\n🤖 Chat Ollama iniciado com modelo '{model}' (local).")
    print("💬 Digite 'fim' para encerrar.\n")

    while True:
        msg = input("Você: ")
        if msg.lower() == "fim":
            break

        url = "http://localhost:11434/api/generate"
        payload = {"model": model, "prompt": msg}

        try:
            response = requests.post(url, json=payload, stream=True)
        except requests.exceptions.ConnectionError:
            print("❌ Erro: o servidor Ollama não está ativo. Execute 'ollama serve' ou abra o Ollama Desktop.")
            break

        resposta_completa = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    print(data["response"], end="", flush=True)
                    resposta_completa += data["response"]
        print("\n")

# ============================================================
# 5️⃣ Menu de seleção de modo
# ============================================================
def main():
    print("="*60)
    print("🤖 Chat Unificado - Hugging Face + Ollama")
    print("="*60)
    print("Escolha o modo de uso:")
    print("1️⃣  Hugging Face (nuvem)")
    print("2️⃣  Ollama (local/offline)")
    print("3️⃣  Sair")
    print("="*60)

    opcao = input("Digite sua opção: ").strip()

    if opcao == "1":
        print("\nModelos Hugging Face recomendados:")
        print(" - microsoft/Phi-3-mini-4k-instruct")
        print(" - meta-llama/Llama-3-8B-Instruct")
        print(" - mistralai/Mistral-7B-Instruct-v0.2")
        print(" - google/gemma-2b-it\n")
        modelo = input("Digite o nome do modelo (ou Enter para padrão): ").strip() or "microsoft/Phi-3-mini-4k-instruct"
        chat_huggingface(model_name=modelo)

    elif opcao == "2":
        print("\nModelos Ollama disponíveis (exemplo):")
        print(" - phi3")
        print(" - mistral")
        print(" - llama3")
        print(" - gemma2")
        print(" - qwen2\n")
        modelo = input("Digite o nome do modelo Ollama (ou Enter para 'phi3'): ").strip() or "phi3"
        chat_ollama(model=modelo)

    else:
        print("👋 Encerrando o chat. Até logo!")
        sys.exit()

if __name__ == "__main__":
    main()