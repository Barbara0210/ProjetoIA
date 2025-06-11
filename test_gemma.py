import keras
import keras_hub
import numpy as np

# Caminho para o modelo local (ajusta se for diferente)
model_path = "models\\gemma3-keras-gemma3_instruct_1b-v3"

# Carregar o modelo localmente
model = keras_hub.models.Gemma3CausalLM.from_preset(model_path)

# Prompt de teste
prompt = "Olá! Que cursos posso tirar na UFP?"

# Tokenizar input
tokens = model.tokenizer.encode(prompt)
tokens = np.array(tokens)[None, :]  # adicionar batch dimension

# Gerar resposta
output_tokens = model.generate(tokens, max_length=100)
response = model.tokenizer.decode(output_tokens[0])

print("Resposta do modelo:")
print(response)
