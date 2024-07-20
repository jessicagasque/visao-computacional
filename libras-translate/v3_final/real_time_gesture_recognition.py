import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms

# Definir o modelo CNN (mesma definição usada para treinar)
class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super(GestureRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*6*6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Assumindo 10 classes de gestos

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 64*6*6)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Carregar o modelo treinado
model = GestureRecognitionModel()
model.load_state_dict(torch.load('gesture_recognition_model.pth'))
model.eval()

# Transformações para os dados
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Função para pré-processar a imagem
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0)
    return image

# Capturar vídeo da webcam
cap = cv2.VideoCapture(0)

# Dicionário de classes (exemplo)
classes = {0: 'Classe1', 1: 'Classe2', 2: 'classe3', 9: 'Classe10'}

while True:
    success, frame = cap.read()
    if not success:
        break

    # Pré-processar o quadro
    input_tensor = preprocess(frame)
    
    # Classificar o gesto
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs.data, 1)
        class_name = classes[predicted.item()]

    # Mostrar a classe predita na imagem
    cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Real-Time Gesture Recognition", frame)

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
