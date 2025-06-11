# Importa os módulos necessários da biblioteca Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datetime import datetime

# Define o ID do modelo que vais usar (modelo pré-treinado da Google chamado Gemma 2B IT)
model_id = "google/gemma-2b-it"
# Carrega o tokenizer (responsável por converter texto em tokens numéricos)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Carrega o modelo de linguagem pré-treinado para tarefas de geração de texto
model = AutoModelForCausalLM.from_pretrained(model_id)

# Cria uma pipeline de geração de texto com o modelo e tokenizer carregados
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

#  Abre o ficheiro com exemplos (few-shot prompting)
# Este ficheiro deve conter exemplos reais de perguntas e respostas sobre a UFP
with open("exemplos_ufp_prompt.txt", "r", encoding="utf-8") as f:
    exemplos = f.read()

#  Cria o nome do ficheiro de log, com a data e hora atual
# Isto permite guardar um histórico da conversa automaticamente
log_filename = f"conversa_ufp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

print("✅ Modelo pronto! Podes escrever perguntas sobre os cursos da UFP.")
print("✏️ Escreve 'sair' para terminar.\n")

#  Ciclo principal do chatbot (loop infinito até o utilizador escrever 'sair')
while True:
    pergunta = input("📩 Tu: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        print("👋 Até à próxima!")
        break

    #  Cria o prompt que será enviado ao modelo
    # Combina os exemplos com a nova pergunta    
    prompt = f"""{exemplos}

Pergunta: {pergunta}
Resposta:"""
    # Usa o modelo para gerar uma resposta com base no prompt
    # Limita a resposta a 150 tokens e permite amostragem para respostas mais variadas
    resposta = generator(prompt, max_new_tokens=150, do_sample=True)[0]["generated_text"]
    # Extrai a resposta final do texto gerado, removendo a parte do prompt
    resposta_final = resposta.split("Resposta:")[-1].strip()

    print("\n🤖 Gemma:\n" + resposta_final + "\n")

    # Guarda pergunta e resposta no ficheiro de log
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(f"Pergunta: {pergunta}\n")
        log_file.write(f"Resposta: {resposta_final}\n\n")
