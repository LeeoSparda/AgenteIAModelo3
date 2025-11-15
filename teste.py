import requests
import json

url = "http://localhost:11434/api/generate"
payload = {"model": "phi3", "prompt": "Olá, tudo bem?"}

print("🔄 Enviando requisição para Ollama...")
response = requests.post(url, json=payload, stream=True)

if response.status_code != 200:
    print(f"❌ Erro HTTP {response.status_code}: {response.text}")
else:
    print("✅ Conectado! Resposta do Ollama:\n")
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                print(data["response"], end="", flush=True)
    print("\n\n✅ Fim da resposta.")
