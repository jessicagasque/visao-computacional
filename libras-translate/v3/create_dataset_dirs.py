import os

dataset_path = "dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# Gestos que você deseja capturar
gestures = ["gesto1", "gesto2", "gesto3"]  # Adicione mais gestos conforme necessário

# Criar pastas para treinamento e teste
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Criar subpastas para cada gesto
for gesture in gestures:
    os.makedirs(os.path.join(train_path, gesture), exist_ok=True)
    os.makedirs(os.path.join(test_path, gesture), exist_ok=True)

print("Estrutura de diretórios criada.")
