from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch

#  Modelo base a utilizar: Gemma 2B Instruct (ajustado para instruções/conversas)
model_id = "google/gemma-2b-it"

# Carregar o tokenizer e o modelo base da HuggingFace
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Configurar o LoRA (Low-Rank Adaptation)
# - `r`: dimensão do rank reduzido
# - `lora_alpha`: fator de escala para os pesos LoRA
# - `target_modules`: camadas onde aplicar LoRA (dependente do modelo)
# - `lora_dropout`: probabilidade de dropout nas camadas LoRA
# - `task_type`: tipo de tarefa — aqui é geração causal de texto
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], # para Gemma, estas são as camadas ajustáveis
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
# Aplica a configuração LoRA ao modelo
model = get_peft_model(model, lora_config)

# Carregar o dataset de treino a partir de um ficheiro JSON com pares pergunta/resposta
dataset = load_dataset("json", data_files="exemplos_ufp_lora.json", split="train")

# Função de tokenização: transforma o texto em tokens para o modelo
def tokenize(sample):
    return tokenizer(sample["text"], truncation=True, padding="max_length", max_length=512)

# Aplica a tokenização ao dataset
tokenized_dataset = dataset.map(tokenize)

# Configurar o data collator que trata do batching e da máscara de atenção
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Configurar os argumentos de treino
training_args = TrainingArguments(
    output_dir="./lora_ufp",  # pasta onde guardar checkpoints
    num_train_epochs=5, # número de épocas de treino
    per_device_train_batch_size=4, # tamanho do batch por dispositivo
    logging_steps=10,  # frequência de logs
    save_strategy="epoch", # salvar modelo ao fim de cada época
    save_total_limit=2, # manter no máximo 2 checkpoints
    learning_rate=2e-4, # taxa de aprendizagem
    fp16=False, # usa FP32 por causa de CPU
)

# Criar o Trainer, que junta tudo para gerir o processo de treino
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Iniciar treino
trainer.train()

# Guardar os pesos LoRA ajustados e o tokenizer
# - `.base_model.save_pretrained(...)` salva o modelo base original
# - `.save_pretrained(...)` salva os deltas e config do LoRA
model.base_model.save_pretrained("modelo_gemma_ufp_lora") 
model.save_pretrained("modelo_gemma_ufp_lora")              
tokenizer.save_pretrained("modelo_gemma_ufp_lora")


