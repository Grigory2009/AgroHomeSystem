# В терминале установите библиотеку:
# pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2

# Загружаем самую большую и точную модель (RTX 5070 с ней справится шутя)
model = YOLO("yolo11x.pt") 

# Открываем поток с веб-камеры (0 - встроенная камера)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Прогоняем кадр через нейросеть
        results = model(frame)
        
        # Отрисовываем результаты на кадре
        annotated_frame = results[0].plot()
        
        # Показываем окно
        cv2.imshow("YOLO Real-Time", annotated_frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()