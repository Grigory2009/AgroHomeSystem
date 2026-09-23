import cv2
import math
from collections import deque, Counter
from pathlib import Path
import sys
import importlib.util

# Загрузка MediaPipe "solutions" модулей напрямую по файлу чтобы избежать импорта Tasks API
# (Tasks импортирует TensorFlow и может конфликтовать с локальными пакетами).
def load_solution_module(module_name):
    # Найти папку site-packages с пакетом mediapipe
    mp_base = None
    for p in sys.path:
        candidate = Path(p) / 'mediapipe' / 'python' / 'solutions' / (module_name + '.py')
        if candidate.exists():
            mp_base = candidate
            break
    if not mp_base:
        raise ImportError(f'Cannot find mediapipe solution module {module_name}')
    spec = importlib.util.spec_from_file_location(f'mediapipe.python.solutions.{module_name}', str(mp_base))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mp_hands = load_solution_module('hands')
mp_face_mesh = load_solution_module('face_mesh')
mp_drawing = load_solution_module('drawing_utils')

# Индексы кончиков пальцев (за исключением большого пальца)
tips_ids = [8, 12, 16, 20]

# Полезные индексы FaceMesh для эвристик
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

LEFT_EYEBROW = 105
RIGHT_EYEBROW = 334

# Подключаемся к веб-камере
cap = cv2.VideoCapture(0)

# Параметры сглаживания выражений
EXPR_HISTORY_LEN = 10
expr_history = deque(maxlen=EXPR_HISTORY_LEN)

# Настраиваем модели MediaPipe (используем классы из загруженных модулей)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2) as hands, \
     mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        img_h, img_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hands_results = hands.process(image_rgb)
        face_results = face_mesh.process(image_rgb)

        gestures_to_draw = []
        expressions_to_draw = []

        # Обработка рук (как раньше)
        if hands_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                hand_label = None
                if hands_results.multi_handedness and len(hands_results.multi_handedness) > idx:
                    hand_label = hands_results.multi_handedness[idx].classification[0].label

                fingers = []
                for i in range(0, 4):
                    tip_id = tips_ids[i]
                    if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                thumb_is_open = False
                try:
                    if hand_label == 'Right':
                        thumb_is_open = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
                    elif hand_label == 'Left':
                        thumb_is_open = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
                    else:
                        thumb_is_open = abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[3].x) > 0.03
                except Exception:
                    thumb_is_open = False

                total_fingers_without_thumb = fingers.count(1)
                total_fingers = total_fingers_without_thumb + (1 if thumb_is_open else 0)

                x_thumb = hand_landmarks.landmark[4].x
                y_thumb = hand_landmarks.landmark[4].y
                x_index = hand_landmarks.landmark[8].x
                y_index = hand_landmarks.landmark[8].y
                dist_thumb_index = math.hypot(x_thumb - x_index, y_thumb - y_index)

                hand_size = math.hypot(
                    hand_landmarks.landmark[0].x - hand_landmarks.landmark[9].x,
                    hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y,
                )

                gesture = 'Unknown'
                if hand_size > 0 and dist_thumb_index / hand_size < 0.35:
                    gesture = 'OK'
                elif thumb_is_open and total_fingers_without_thumb == 0:
                    gesture = 'Thumbs Up'
                elif fingers == [1, 1, 0, 0]:
                    gesture = 'Peace'
                elif fingers == [1, 0, 0, 0]:
                    gesture = 'Pointing'
                elif total_fingers == 5:
                    gesture = 'Open Palm'
                elif total_fingers == 0:
                    gesture = 'Fist'
                else:
                    gesture = f'{total_fingers} fingers'

                wrist_x = int(hand_landmarks.landmark[0].x * img_w)
                wrist_y = int(hand_landmarks.landmark[0].y * img_h)
                gestures_to_draw.append((gesture, wrist_x, wrist_y))

        # Обработка лица: эвристики + сглаживание по истории
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1))

                def lm_xy(i):
                    lm = face_landmarks.landmark[i]
                    return (lm.x * img_w, lm.y * img_h)

                ml_x, ml_y = lm_xy(MOUTH_LEFT)
                mr_x, mr_y = lm_xy(MOUTH_RIGHT)
                mt_x, mt_y = lm_xy(MOUTH_TOP)
                mb_x, mb_y = lm_xy(MOUTH_BOTTOM)

                # Ширина/высота рта
                mouth_width = math.hypot(mr_x - ml_x, mr_y - ml_y)
                mouth_height = math.hypot(mb_x - mt_x, mb_y - mt_y)

                # Нормализуем по расстоянию между глазами
                le_top_x, le_top_y = lm_xy(LEFT_EYE_TOP)
                re_top_x, re_top_y = lm_xy(RIGHT_EYE_TOP)
                eye_dist = math.hypot(re_top_x - le_top_x, re_top_y - le_top_y)

                smile_ratio = mouth_width / (mouth_height + 1e-6)
                mouth_open_ratio = mouth_height / (eye_dist + 1e-6)

                # Sadness: проверяем опущенные уголки рта (координата Y больше чем центр)
                mouth_center_y = (mt_y + mb_y) / 2
                corners_down = ((ml_y - mouth_center_y) > 3) and ((mr_y - mouth_center_y) > 3)

                # Глаза
                l_et_x, l_et_y = lm_xy(LEFT_EYE_TOP)
                l_eb_x, l_eb_y = lm_xy(LEFT_EYE_BOTTOM)
                r_et_x, r_et_y = lm_xy(RIGHT_EYE_TOP)
                r_eb_x, r_eb_y = lm_xy(RIGHT_EYE_BOTTOM)

                left_eye_open = (abs(l_et_y - l_eb_y) / (eye_dist + 1e-6)) > 0.02
                right_eye_open = (abs(r_et_y - r_eb_y) / (eye_dist + 1e-6)) > 0.02

                # Брови
                le_yb_x, le_yb_y = lm_xy(LEFT_EYEBROW)
                re_yb_x, re_yb_y = lm_xy(RIGHT_EYEBROW)
                brow_left_gap = (le_yb_y - l_et_y) / (eye_dist + 1e-6)
                brow_right_gap = (re_yb_y - r_et_y) / (eye_dist + 1e-6)

                expression = 'Neutral'
                # Улучшенные правила (пороговые значения можно подогнать)
                if smile_ratio > 6.5 and mouth_open_ratio < 0.25:
                    expression = 'Smile'
                elif corners_down and not left_eye_open and not right_eye_open:
                    expression = 'Sad'
                elif mouth_open_ratio > 0.45 and (brow_left_gap > 0.12 or brow_right_gap > 0.12):
                    expression = 'Surprised'
                elif mouth_open_ratio > 0.4:
                    expression = 'Mouth Open'
                elif not left_eye_open and not right_eye_open:
                    expression = 'Eyes Closed'
                elif (brow_left_gap > 0.12 or brow_right_gap > 0.12) and mouth_open_ratio < 0.18:
                    expression = 'Raised Eyebrows'

                # Добавляем в историю и выбираем наиболее частое за окно
                expr_history.append(expression)
                most_common = Counter(expr_history).most_common(1)[0][0]

                # Координаты для отрисовки (над лбом)
                xs = [p.x for p in face_landmarks.landmark]
                ys = [p.y for p in face_landmarks.landmark]
                x_min = int(max(min(xs) * img_w - 20, 0))
                x_max = int(min(max(xs) * img_w + 20, img_w))
                y_min = int(max(min(ys) * img_h - 20, 0))
                forehead_x = int((x_min + x_max) / 2)
                forehead_y = int(y_min - 20)

                expressions_to_draw.append((most_common, forehead_x, forehead_y))

        # Отражаем итоговое изображение зеркально
        image = cv2.flip(image, 1)

        # Рисуем жесты
        for gesture, wx, wy in gestures_to_draw:
            wx = img_w - wx
            cv2.putText(image, gesture, (wx - 30, wy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

        cv2.imshow('Cyber Gestures & Expressions', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
