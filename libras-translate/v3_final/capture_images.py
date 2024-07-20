import cv2
import os

# Modo de captura (train ou test)
mode = "train"  # Mude para "test" quando capturar imagens de teste
dataset_path = f"dataset/{mode}"

# Gestos que você deseja capturar
gestures = ["gesto1", "gesto2", "gesto3"]  # Adicione mais gestos conforme necessário

# Criar pastas para cada gesto
for gesture in gestures:
    os.makedirs(os.path.join(dataset_path, gesture), exist_ok=True)

cap = cv2.VideoCapture(0)
current_gesture = 0

while current_gesture < len(gestures):
    gesture_name = gestures[current_gesture]
    count = len(os.listdir(os.path.join(dataset_path, gesture_name)))

    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Gesture: {gesture_name}, Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Capture Gestures", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        img_path = os.path.join(dataset_path, gesture_name, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
    elif key == ord('n'):
        current_gesture += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
