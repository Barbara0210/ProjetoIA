from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from datetime import datetime
import faiss

#  Definir o ID do modelo base Gemma e o nome do ficheiro de log
model_id = "google/gemma-2b-it"
log_filename = f"conversa_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

#  Carregar modelo de embeddings semânticos (usado para comparar texto por significado)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#  Carregar base de conhecimento (documentos) do ficheiro JSON
dataset = load_dataset("json", data_files="documentos_ufp.json")["train"]
documentos = [d["text"] for d in dataset]

#  Criar embeddings dos documentos e construir índice FAISS para busca eficiente
document_embeddings = embedder.encode(documentos)
index = faiss.IndexFlatL2(document_embeddings[0].shape[0])
index.add(document_embeddings)

#  Carregar modelo de linguagem (LLM) e tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("✅ RAG Chatbot pronto! Escreve perguntas sobre a UFP.")
print("✏️ Escreve 'sair' para terminar.\n")
# 🔄 Iniciar loop de conversa com o utilizador
while True:
    pergunta = input("📩 Tu: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        print("👋 Até à próxima!")
        break

    # Gerar embedding da pergunta e recuperar os 3 documentos mais semelhantes
    pergunta_emb = embedder.encode([pergunta])
    top_k = 3
    _, indices = index.search(pergunta_emb, top_k)
    contexto = "\n\n".join([documentos[i] for i in indices[0]])

    # Criar o prompt a ser enviado ao modelo, com instruções e contexto
    prompt = f"""
Tu és um assistente da Universidade Fernando Pessoa (UFP).
Usa apenas o seguinte contexto para responder à pergunta do utilizador de forma clara, correta e concisa.

Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:"""

    # Gerar resposta
    resposta = generator(prompt, max_new_tokens=150, do_sample=True)[0]["generated_text"]
    resposta_final = resposta.split("Resposta:")[-1].strip()

    print("\n🤖 Gemma:\n" + resposta_final + "\n")

    #  Guardar pergunta, contexto e resposta no ficheiro de log
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(f"Pergunta: {pergunta}\n")
        log_file.write(f"Contexto usado:\n{contexto}\n")
        log_file.write(f"Resposta:\n{resposta_final}\n\n")
