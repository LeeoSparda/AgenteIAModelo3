import os
import subprocess

modelos = [
    "phi3",
    "llama3",
    "mistral",
    "gemma",
    "moondream",
    "llava",
    "neural-chat",
    "codellama"
]

for modelo in modelos:
    print(f"🔽 Baixando modelo {modelo} ...")
    resultado = subprocess.run(["ollama", "pull", modelo])
    if resultado.returncode == 0:
        print(f"✅ Modelo {modelo} baixado com sucesso.\n")
    else:
        print(f"❌ Erro ao baixar o modelo {modelo}.\n")