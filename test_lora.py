from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig
import torch

# 1. Caminho do modelo com LoRA afinado
lora_model_path = "modelo_gemma_ufp_lora"

# 2. Carregar config LoRA
config = PeftConfig.from_pretrained(lora_model_path)

# 3. Carregar modelo base + aplicar LoRA
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, lora_model_path)

# 4. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

# 5. Criar pipeline com parâmetros seguros
chat = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")

print("✅ Modelo LoRA carregado. Escreve a tua pergunta.")
print("✏️ Escreve 'sair' para terminar.\n")

# 6. Loop de conversa
while True:
    pergunta = input("📩 Tu: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        print("👋 Até à próxima!")
        break

    prompt = f"Pergunta: {pergunta}\nResposta:"

    try:
        resposta = chat(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]
        resposta_final = resposta.split("Resposta:")[-1].strip()
        print("\n🤖 Gemma LoRA:\n" + resposta_final + "\n")

    except Exception as e:
        print(f"❌ Erro ao gerar resposta: {str(e)}\n")
