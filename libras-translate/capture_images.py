import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Diretório onde as imagens serão salvas
dataset_path = "dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# Gestos que você deseja capturar
gestures = ["gesto1", "gesto2", "gesto3"]  # Adicione mais gestos conforme necessário

# Criar pastas para cada gesto em train e test
for gesture in gestures:
    os.makedirs(os.path.join(train_path, gesture), exist_ok=True)
    os.makedirs(os.path.join(test_path, gesture), exist_ok=True)

cap = cv2.VideoCapture(0)
current_gesture = 0

# Lista para armazenar todas as imagens capturadas temporariamente
all_images = {gesture: [] for gesture in gestures}

while current_gesture < len(gestures):
    gesture_name = gestures[current_gesture]

    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Gesture: {gesture_name}, Press 'c' to capture", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Press 'n' for next gesture, 'q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Capture Gestures", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        all_images[gesture_name].append(frame.copy())
        print(f"Captured image for {gesture_name}")
    elif key == ord('n'):
        current_gesture += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Split images into training and testing sets (80% train, 20% test)
for gesture, images in all_images.items():
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
    for i, img in enumerate(train_images):
        img_path = os.path.join(train_path, gesture, f"{i}.jpg")
        cv2.imwrite(img_path, img)
    for i, img in enumerate(test_images):
        img_path = os.path.join(test_path, gesture, f"{i}.jpg")
        cv2.imwrite(img_path, img)

print("Images saved in 'dataset/train' and 'dataset/test' directories.")
