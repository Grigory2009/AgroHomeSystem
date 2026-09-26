"""
AgroHomeSystem - Per-Cell Plant Tracking & Matrix Calibration Engine
=============================================================================
Модуль индивидуального поячеечного трекинга для 3D-корпуса AgroHomeSystem v2
(лоток на 24 ячейки: сетка 4x6 под минераловатные пробки 22 мм) с поддержкой:
1. Распознавания желтых минераловатных пробок (22 мм Rockwool plugs).
2. Автоматического обнаружения лотка и посадочных пробок (Computer Vision).
3. Разделения спектра: Минвата (желтый субстрат) vs Листва vs Хлороз vs Плесень.
4. Ручной калибровки геометрии матрицы (ROI, радиусы, ряды/колонки).
5. Индивидуального ручного редактирования ячеек (культура, статус, дата, заметки).
6. Сохранения и загрузки конфигурации в JSON (cell_grid_config.json).
=============================================================================
"""

import os
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np


@dataclass
class CellMetrics:
    """Биометрические показатели отдельной ячейки лотка."""
    cell_id: str                      # Например: "A1", "A2" ... "F4"
    index: int                        # 0..23
    row: int                          # 0..5
    col: int                          # 0..3
    center_x: int                     # Центр ячейки в пикселях кадра
    center_y: int
    radius: int                       # Радиус посадочного гнезда 22 мм
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    
    total_pixels: int = 0             # Всего пикселей внутри маски ячейки
    foliage_pixels: int = 0           # Пикселей растительной биомассы
    healthy_green_pixels: int = 0     # Пикселей сочного зеленого хлорофилла
    chlorosis_pixels: int = 0         # Пикселей истинного пожелтения/хлороза листвы
    necrosis_pixels: int = 0          # Пикселей отмершей ткани
    mold_pixels: int = 0              # Пикселей мицелия плесени
    rockwool_pixels: int = 0          # Пикселей желтой минеральной ваты (субстрат)
    
    coverage_ratio: float = 0.0       # 0.0 .. 1.0 (площадь покрытия ячейки растением)
    chlorosis_ratio: float = 0.0      # Доля хлороза от биомассы
    necrosis_ratio: float = 0.0       # Доля некроза от биомассы
    mold_ratio: float = 0.0           # Доля мицелия от ячейки
    rockwool_ratio: float = 0.0       # Доля желтой минваты в гнезде
    mean_exg: float = 0.0             # Средний индекс Excess Green
    
    health_index: float = 100.0       # 0.0 .. 100.0 %
    stage: str = "EMPTY"              # "EMPTY", "ROCKWOOL", "GERMINATING", "SPROUT", "MATURE"
    status: str = "EMPTY"             # "HEALTHY", "WARNING", "CRITICAL", "ROCKWOOL", "EMPTY"
    diagnosis_ru: str = "Свободно"
    diagnosis_en: str = "Empty"
    
    # Пользовательские поля и ручные переопределения
    crop_name: str = "Микрозелень"
    planted_date: str = ""
    notes: str = ""
    is_manual_override: bool = False
    
    last_updated: float = field(default_factory=time.time)


class GridCellTracker:
    """
    Трекер индивидуальных ячеек сетки посадочного лотка.
    По умолчанию настроен на переработанный корпус AgroHomeSystem v2:
    Сетка 4 колонки x 6 рядов (24 ячейки диаметром 22 мм).
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 4,
        roi_norm: Tuple[float, float, float, float] = (0.06, 0.08, 0.94, 0.92),
        cell_radius_ratio: float = 0.38,
        config_path: Optional[str] = None
    ):
        """
        :param rows: Количество рядов (по умолчанию 6)
        :param cols: Количество колонок (по умолчанию 4)
        :param roi_norm: Относительные границы лотка в кадре (ymin, xmin, ymax, xmax) в диапазоне 0.0..1.0
        :param cell_radius_ratio: Радиус окружности ячейки относительно шага сетки
        :param config_path: Путь к файлу конфигурации JSON
        """
        self.rows = max(1, min(rows, 12))
        self.cols = max(1, min(cols, 12))
        self.roi_norm = roi_norm
        self.cell_radius_ratio = max(0.1, min(cell_radius_ratio, 0.5))
        
        # Словарь пользовательских настроек для конкретных ячеек (cell_id -> {crop_name, planted_date, status_override, notes})
        self.cell_overrides: Dict[str, Dict[str, Any]] = {}
        
        # Определение пути конфигурационного файла
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path(__file__).resolve().parent / "cell_grid_config.json"

        # История динамики прироста биомассы по ячейкам
        self.history: Dict[str, List[Tuple[float, float]]] = {}
        self.last_metrics: List[CellMetrics] = []
        
        # Автоматическая загрузка сохраненной конфигурации при старте
        self.load_config()

    def set_roi(self, ymin: float, xmin: float, ymax: float, xmax: float):
        """Калибровка границ лотка в поле зрения камеры."""
        self.roi_norm = (
            max(0.0, min(float(ymin), 0.85)),
            max(0.0, min(float(xmin), 0.85)),
            min(1.0, max(float(ymax), 0.15)),
            min(1.0, max(float(xmax), 0.15))
        )

    def set_geometry(self, rows: int, cols: int, roi_norm: Tuple[float, float, float, float], radius_ratio: float):
        """Полная установка геометрических параметров матрицы."""
        self.rows = max(1, min(int(rows), 12))
        self.cols = max(1, min(int(cols), 12))
        self.set_roi(*roi_norm)
        self.cell_radius_ratio = max(0.1, min(float(radius_ratio), 0.5))

    def set_cell_override(
        self,
        cell_id: str,
        crop_name: Optional[str] = None,
        planted_date: Optional[str] = None,
        status_override: Optional[str] = None,
        notes: Optional[str] = None
    ):
        """Установить ручные параметры для конкретной ячейки."""
        cid = str(cell_id).upper().strip()
        if cid not in self.cell_overrides:
            self.cell_overrides[cid] = {}
        
        entry = self.cell_overrides[cid]
        if crop_name is not None:
            entry["crop_name"] = str(crop_name).strip()
        if planted_date is not None:
            entry["planted_date"] = str(planted_date).strip()
        if status_override is not None:
            entry["status_override"] = str(status_override).strip().upper()
        if notes is not None:
            entry["notes"] = str(notes).strip()
            
        self.save_config()

    def clear_cell_override(self, cell_id: str):
        """Сбросить ручные параметры ячейки к автоматическим."""
        cid = str(cell_id).upper().strip()
        if cid in self.cell_overrides:
            del self.cell_overrides[cid]
            self.save_config()

    def save_config(self, filepath: Optional[str] = None) -> bool:
        """Сохранение калибровки и настроек матрицы в JSON-файл."""
        path = Path(filepath) if filepath else self.config_path
        data = {
            "version": "2.0",
            "tray_layout": f"{self.cols}x{self.rows}",
            "rows": self.rows,
            "cols": self.cols,
            "roi_norm": list(self.roi_norm),
            "cell_radius_ratio": round(float(self.cell_radius_ratio), 3),
            "cell_overrides": self.cell_overrides,
            "last_saved": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[GridCellTracker] Ошибка сохранения конфига в {path}: {e}")
            return False

    def load_config(self, filepath: Optional[str] = None) -> bool:
        """Загрузка калибровки матрицы из JSON-файла."""
        path = Path(filepath) if filepath else self.config_path
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "rows" in data:
                self.rows = max(1, min(int(data["rows"]), 12))
            if "cols" in data:
                self.cols = max(1, min(int(data["cols"]), 12))
            if "roi_norm" in data and len(data["roi_norm"]) == 4:
                self.set_roi(*data["roi_norm"])
            if "cell_radius_ratio" in data:
                self.cell_radius_ratio = max(0.1, min(float(data["cell_radius_ratio"]), 0.5))
            if "cell_overrides" in data and isinstance(data["cell_overrides"], dict):
                self.cell_overrides = data["cell_overrides"]
            return True
        except Exception as e:
            print(f"[GridCellTracker] Ошибка чтения конфига из {path}: {e}")
            return False

    def autodetect_tray(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Интеллектуальное автообнаружение лотка и желтых пробок минеральной ваты (22 мм):
        1. Спектральная детекция желтого волокнистого субстрата минваты (Rockwool CV).
        2. Поиск центров пробок через контурный анализ и Hough Circles.
        3. Оценка геометрического охвата (ROI) по найденным пробкам или контуру лотка.
        :param frame_bgr: Кадр с камеры
        :return: Словарь с результатами детекции и новым ROI
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {"status": "error", "message": "Пустой кадр", "roi": list(self.roi_norm)}

        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # 1. Специфическая маска желтой минеральной ваты (Rockwool plug detection)
        # Желтый/соломенный/песочный оттенок субстрата: H 18..46, S 25..200, V 70..255
        lower_wool = np.array([16, 25, 60], dtype=np.uint8)
        upper_wool = np.array([48, 220, 255], dtype=np.uint8)
        raw_wool_mask = cv2.inRange(hsv, lower_wool, upper_wool)
        
        # Морфологическая очистка волокнистой текстуры ваты
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        wool_clean = cv2.morphologyEx(raw_wool_mask, cv2.MORPH_CLOSE, kernel)
        wool_clean = cv2.morphologyEx(wool_clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # 2. Поиск круглых/цилиндрических пробок минваты по контурам
        detected_plugs = []
        min_rad = max(8, int(min(w, h) * 0.02))
        max_rad = max(min_rad + 8, int(min(w, h) * 0.16))
        min_dist = max(16, int(min(w, h) * 0.06))

        contours, _ = cv2.findContours(wool_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (math.pi * min_rad * min_rad * 0.4):
                (cx, cy), rad = cv2.minEnclosingCircle(cnt)
                cx, cy, rad = int(cx), int(cy), int(rad)
                if min_rad <= rad <= max_rad:
                    if 0.03 * w < cx < 0.97 * w and 0.03 * h < cy < 0.97 * h:
                        detected_plugs.append((cx, cy, rad))

        # 3. Дополнительный поиск Hough Circles по маске минваты и по серому кадру
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2.0)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=70,
            param2=30,
            minRadius=min_rad,
            maxRadius=max_rad
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                if 0.04 * w < cx < 0.96 * w and 0.04 * h < cy < 0.96 * h:
                    # Проверяем, что в этой области есть желтая минвата или посадочное отверстие
                    patch = wool_clean[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)]
                    if patch.size > 0 and np.mean(patch) > 20:
                        if not any(math.hypot(cx - dp[0], cy - dp[1]) < min_dist * 0.7 for dp in detected_plugs):
                            detected_plugs.append((cx, cy, r))

        detected_count = len(detected_plugs)
        new_roi = list(self.roi_norm)
        confidence = 50.0
        method = "default"

        if detected_count >= 2:
            # Найдена группа желтых пробок минваты: строим охватывающий прямоугольник с отступом
            xs = [p[0] for p in detected_plugs]
            ys = [p[1] for p in detected_plugs]
            avg_r = float(np.mean([p[2] for p in detected_plugs]))

            # Отступ в размере радиуса пробки
            min_x = max(0, min(xs) - int(avg_r * 1.6))
            max_x = min(w, max(xs) + int(avg_r * 1.6))
            min_y = max(0, min(ys) - int(avg_r * 1.6))
            max_y = min(h, max(ys) + int(avg_r * 1.6))

            new_ymin = round(min_y / float(h), 3)
            new_xmin = round(min_x / float(w), 3)
            new_ymax = round(max_y / float(h), 3)
            new_xmax = round(max_x / float(w), 3)

            if 0.10 < (new_xmax - new_xmin) < 0.98 and 0.10 < (new_ymax - new_ymin) < 0.98:
                self.set_roi(new_ymin, new_xmin, new_ymax, new_xmax)
                new_roi = list(self.roi_norm)
                confidence = min(99.0, 65.0 + detected_count * 2.5)
                method = "yellow_rockwool_plugs"

        else:
            # Попытка детекции внешнего прямоугольного контура лотка
            _, bin_tray = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            tray_contours, _ = cv2.findContours(bin_tray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_rect = None
            max_area = 0

            for cnt in tray_contours:
                area = cv2.contourArea(cnt)
                if area > (w * h * 0.15):
                    x, y, tw, th = cv2.boundingRect(cnt)
                    if area > max_area:
                        max_area = area
                        best_rect = (x, y, tw, th)

            if best_rect:
                x, y, tw, th = best_rect
                pad_x = int(tw * 0.05)
                pad_y = int(th * 0.05)
                new_ymin = round(max(0, y + pad_y) / float(h), 3)
                new_xmin = round(max(0, x + pad_x) / float(w), 3)
                new_ymax = round(min(h, y + th - pad_y) / float(h), 3)
                new_xmax = round(min(w, x + tw - pad_x) / float(w), 3)
                self.set_roi(new_ymin, new_xmin, new_ymax, new_xmax)
                new_roi = list(self.roi_norm)
                confidence = 80.0
                method = "tray_outer_contour"

        # Сохраняем обновленные координаты
        self.save_config()

        return {
            "status": "ok",
            "method": method,
            "detected_holes": detected_count,
            "detected_rockwool": detected_count,
            "confidence": round(confidence, 1),
            "roi": new_roi,
            "rows": self.rows,
            "cols": self.cols,
            "cell_radius_ratio": self.cell_radius_ratio,
            "message": f"Распознано {detected_count} пробок минваты (22 мм). Метод: {method}. ROI обновлен."
        }

    def _generate_grid_geometry(self, frame_w: int, frame_h: int) -> List[Dict[str, Any]]:
        """Расчет координат центров и радиусов 24 ячеек с учетом геометрии кадра."""
        ymin, xmin, ymax, xmax = self.roi_norm
        tray_x1 = int(xmin * frame_w)
        tray_y1 = int(ymin * frame_h)
        tray_x2 = int(xmax * frame_w)
        tray_y2 = int(ymax * frame_h)

        tray_w = max(10, tray_x2 - tray_x1)
        tray_h = max(10, tray_y2 - tray_y1)

        step_x = tray_w / float(self.cols)
        step_y = tray_h / float(self.rows)

        radius = int(min(step_x, step_y) * self.cell_radius_ratio)
        radius = max(8, radius)

        row_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        cells_geo = []

        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                cx = int(tray_x1 + (c + 0.5) * step_x)
                cy = int(tray_y1 + (r + 0.5) * step_y)
                row_label = row_letters[r] if r < len(row_letters) else f"R{r+1}"
                cell_id = f"{row_label}{c+1}"

                x1 = max(0, cx - radius)
                y1 = max(0, cy - radius)
                x2 = min(frame_w, cx + radius)
                y2 = min(frame_h, cy + radius)

                cells_geo.append({
                    "cell_id": cell_id,
                    "index": idx,
                    "row": r,
                    "col": c,
                    "cx": cx,
                    "cy": cy,
                    "radius": radius,
                    "bbox": (x1, y1, x2, y2)
                })
                idx += 1

        return cells_geo

    def analyze(
        self,
        frame_bgr: np.ndarray,
        exg_img: Optional[np.ndarray] = None,
        foliage_mask: Optional[np.ndarray] = None
    ) -> List[CellMetrics]:
        """
        Комплексный спектральный анализ всех ячеек на кадре:
        - Четкое разделение желтой минеральной ваты (субстрат) от хлороза листвы.
        - Расчет покрытия пологом (Biomass), хлороза (Chlorosis), плесени (Mold).
        - Учет ручных переопределений и калибровок.
        """
        h, w = frame_bgr.shape[:2]
        now = time.time()

        # 1. Расчет индекса Excess Green (ExG = 2G - R - B)
        b = frame_bgr[:, :, 0].astype(np.float32)
        g = frame_bgr[:, :, 1].astype(np.float32)
        r = frame_bgr[:, :, 2].astype(np.float32)
        exg_raw = 2.0 * g - r - b
        if exg_img is None:
            exg_img = np.clip(exg_raw, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # 2. СПЕКТРАЛЬНЫЕ МАСКИ:
        # А. Желтая минеральная вата (22мм пробки Rockwool):
        # Оттенок H 16..46, умеренная насыщенность S 20..190, яркость V 65..255.
        # ВАЖНО: у сухой/влажной минваты exg_raw < 22 (нет активного хлорофилла!).
        lower_wool = np.array([16, 20, 65], dtype=np.uint8)
        upper_wool = np.array([46, 200, 255], dtype=np.uint8)
        wool_hsv = cv2.inRange(hsv, lower_wool, upper_wool)
        rockwool_mask = np.where((wool_hsv > 0) & (exg_raw < 24), 255, 0).astype(np.uint8)

        # Б. Зеленая активная листва (Хлорофилл):
        # Зеленые оттенки H 32..90, высокая доминанта зеленого exg_img >= 20
        lower_green = np.array([30, 30, 30], dtype=np.uint8)
        upper_green = np.array([92, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_active = np.where((green_mask > 0) & (exg_raw >= 20), 255, 0).astype(np.uint8)

        # В. Хлороз (Патологическое пожелтение листовой пластины):
        # В отличие от минваты, хлороз проявляется на реальных листьях, где есть листовой полог (exg_raw >= 25 и S > 55)
        lower_yellow = np.array([18, 55, 70], dtype=np.uint8)
        upper_yellow = np.array([32, 255, 255], dtype=np.uint8)
        yellow_leaf_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        chlorosis_mask = np.where((yellow_leaf_mask > 0) & (exg_raw >= 25) & (rockwool_mask == 0), 255, 0).astype(np.uint8)

        # Г. Суммарная растительная биомасса (зелень + хлороз листвы, БЕЗ МИНВАТЫ!)
        if foliage_mask is None:
            foliage_mask = cv2.bitwise_or(green_active, chlorosis_mask)
            # Исключаем пиксели минеральной ваты из биомассы
            foliage_mask = np.where((foliage_mask > 0) & (rockwool_mask == 0), 255, 0).astype(np.uint8)

        # Д. Некроз (темно-коричневые / черные пятна отмирания ткани)
        lower_necrosis = np.array([5, 40, 20], dtype=np.uint8)
        upper_necrosis = np.array([24, 255, 85], dtype=np.uint8)
        necrosis_mask = cv2.inRange(hsv, lower_necrosis, upper_necrosis)
        necrosis_mask = np.where((necrosis_mask > 0) & (rockwool_mask == 0), 255, 0).astype(np.uint8)

        # Е. Мицелий серой гнили (Botrytis / плесень: серо-белый налет, низкая насыщенность S < 35, V 110..220)
        lower_mold = np.array([0, 0, 110], dtype=np.uint8)
        upper_mold = np.array([180, 35, 215], dtype=np.uint8)
        mold_candidate = cv2.inRange(hsv, lower_mold, upper_mold)
        # Исключаем чистую минвату и активную зелень
        mold_mask = np.where((mold_candidate > 0) & (exg_img < 15) & (rockwool_mask == 0), 255, 0).astype(np.uint8)

        grid_geo = self._generate_grid_geometry(w, h)
        results: List[CellMetrics] = []

        for item in grid_geo:
            cid = item["cell_id"]
            cx = item["cx"]
            cy = item["cy"]
            rad = item["radius"]

            # Создаем круговую маску ячейки
            cell_circle_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(cell_circle_mask, (cx, cy), rad, 255, -1)

            total_px = int(np.count_nonzero(cell_circle_mask))
            if total_px == 0:
                total_px = 1

            # Пересечение с масками биометрии
            foliage_px = int(np.count_nonzero((foliage_mask > 0) & (cell_circle_mask > 0)))
            green_px = int(np.count_nonzero((green_active > 0) & (cell_circle_mask > 0)))
            chl_px = int(np.count_nonzero((chlorosis_mask > 0) & (cell_circle_mask > 0)))
            nec_px = int(np.count_nonzero((necrosis_mask > 0) & (cell_circle_mask > 0)))
            mld_px = int(np.count_nonzero((mold_mask > 0) & (cell_circle_mask > 0)))
            wool_px = int(np.count_nonzero((rockwool_mask > 0) & (cell_circle_mask > 0)))

            coverage = round(foliage_px / float(total_px), 3)
            chl_ratio = round(chl_px / float(max(1, foliage_px)), 3)
            nec_ratio = round(nec_px / float(max(1, foliage_px)), 3)
            mld_ratio = round(mld_px / float(total_px), 3)
            wool_ratio = round(wool_px / float(total_px), 3)

            # Средний ExG внутри ячейки
            cell_exg_vals = exg_img[cell_circle_mask > 0]
            mean_exg = float(np.mean(cell_exg_vals)) if len(cell_exg_vals) > 0 else 0.0

            # 3. Определение стадии и здоровья ячейки
            stage = "EMPTY"
            status = "EMPTY"
            diag_ru = "Свободно"
            diag_en = "Empty"
            health_idx = 100.0

            if coverage < 0.04 and wool_ratio >= 0.18:
                # В ячейке установлена желтая минеральная вата, семена еще не взошли
                stage = "ROCKWOOL"
                status = "ROCKWOOL"
                diag_ru = "Минвата 22мм (Готово к посеву)"
                diag_en = "22mm Rockwool Plug (Ready)"
                health_idx = 100.0
            elif coverage < 0.04:
                # Пустое гнездо лотка
                stage = "EMPTY"
                status = "EMPTY"
                diag_ru = "Пустая ячейка"
                diag_en = "Empty Cell"
                health_idx = 100.0
            elif mld_ratio >= 0.08:
                stage = "MOLD"
                status = "CRITICAL"
                diag_ru = "Плесень (Серая гниль)"
                diag_en = "Gray Mold (Botrytis)"
                health_idx = max(5.0, 30.0 - mld_ratio * 100.0)
            elif nec_ratio >= 0.18 and coverage > 0.08:
                stage = "DAMPING_OFF"
                status = "CRITICAL"
                diag_ru = "Полегание (Черная ножка)"
                diag_en = "Damping-Off (Pythium)"
                health_idx = max(10.0, 40.0 - nec_ratio * 100.0)
            elif chl_ratio >= 0.25:
                stage = "CHLOROSIS"
                status = "WARNING"
                diag_ru = "Хлороз (дефицит Fe/N)"
                diag_en = "Chlorosis (Fe/N def)"
                health_idx = max(40.0, 80.0 - chl_ratio * 80.0)
            elif coverage < 0.18:
                stage = "GERMINATING"
                status = "HEALTHY"
                diag_ru = "Прорастание (Всходы)"
                diag_en = "Germination (Radicle)"
                health_idx = 96.0
            elif coverage < 0.45:
                stage = "SPROUT"
                status = "HEALTHY"
                diag_ru = "Семядоли (Росток)"
                diag_en = "Cotyledon (Sprout)"
                health_idx = 98.0
            else:
                stage = "MATURE"
                status = "HEALTHY"
                diag_ru = "Спелая микрозелень"
                diag_en = "Mature Canopy"
                health_idx = 100.0

            # Применение пользовательских ручных переопределений
            crop_name = "Микрозелень"
            planted_date = ""
            notes = ""
            is_override = False

            if cid in self.cell_overrides:
                ov = self.cell_overrides[cid]
                crop_name = ov.get("crop_name", crop_name)
                planted_date = ov.get("planted_date", planted_date)
                notes = ov.get("notes", notes)
                
                st_ov = ov.get("status_override")
                if st_ov and st_ov not in ("AUTO", ""):
                    status = st_ov
                    is_override = True
                    if st_ov == "HEALTHY":
                        diag_ru = f"Вручную: Здорово ({crop_name})"
                    elif st_ov == "WARNING":
                        diag_ru = f"Вручную: Требует внимания ({crop_name})"
                    elif st_ov == "CRITICAL":
                        diag_ru = f"Вручную: Критично ({crop_name})"
                    elif st_ov == "ROCKWOOL":
                        diag_ru = "Вручную: Минвата 22мм"
                        stage = "ROCKWOOL"
                    elif st_ov == "EMPTY":
                        diag_ru = "Вручную: Пустая ячейка"
                        stage = "EMPTY"

            metric = CellMetrics(
                cell_id=cid,
                index=item["index"],
                row=item["row"],
                col=item["col"],
                center_x=cx,
                center_y=cy,
                radius=rad,
                bbox=item["bbox"],
                total_pixels=total_px,
                foliage_pixels=foliage_px,
                healthy_green_pixels=green_px,
                chlorosis_pixels=chl_px,
                necrosis_pixels=nec_px,
                mold_pixels=mld_px,
                rockwool_pixels=wool_px,
                coverage_ratio=coverage,
                chlorosis_ratio=chl_ratio,
                necrosis_ratio=nec_ratio,
                mold_ratio=mld_ratio,
                rockwool_ratio=wool_ratio,
                mean_exg=round(mean_exg, 1),
                health_index=round(health_idx, 1),
                stage=stage,
                status=status,
                diagnosis_ru=diag_ru,
                diagnosis_en=diag_en,
                crop_name=crop_name,
                planted_date=planted_date,
                notes=notes,
                is_manual_override=is_override,
                last_updated=now
            )
            results.append(metric)

        self.last_metrics = results
        return results

    def get_summary(self, metrics: Optional[List[CellMetrics]] = None) -> Dict[str, Any]:
        """Сводный статистический отчет по всему лотку."""
        if metrics is None:
            metrics = self.last_metrics

        total = len(metrics)
        empty = sum(1 for m in metrics if m.status == "EMPTY")
        rockwool = sum(1 for m in metrics if m.status == "ROCKWOOL")
        healthy = sum(1 for m in metrics if m.status == "HEALTHY")
        warning = sum(1 for m in metrics if m.status == "WARNING")
        critical = sum(1 for m in metrics if m.status == "CRITICAL")
        active = healthy + warning + critical

        avg_cov = float(np.mean([m.coverage_ratio for m in metrics])) if metrics else 0.0
        avg_health = float(np.mean([m.health_index for m in metrics if m.status in ("HEALTHY", "WARNING", "CRITICAL")])) if active > 0 else 100.0

        return {
            "total_cells": total,
            "active_cells": active,
            "rockwool_cells": rockwool,
            "empty_cells": empty,
            "healthy_cells": healthy,
            "warning_cells": warning,
            "critical_cells": critical,
            "avg_coverage_percent": round(avg_cov * 100.0, 1),
            "avg_health_score": round(avg_health, 1),
            "layout": f"{self.cols}x{self.rows}",
            "rows": self.rows,
            "cols": self.cols,
            "roi_norm": list(self.roi_norm),
            "cell_radius_ratio": self.cell_radius_ratio
        }

    def draw_hud(
        self,
        frame: np.ndarray,
        metrics: Optional[List[CellMetrics]] = None,
        show_summary: bool = True
    ) -> np.ndarray:
        """
        Отрисовка технологичного киберпанк HUD-оверлея на кадре:
        - Дифференциация: Минвата (желтый янтарь), Здорово (изумруд), Хлороз (оранжевый), Болезнь (алый), Пусто (серый).
        """
        if metrics is None:
            metrics = self.last_metrics

        output = frame.copy()
        h, w = output.shape[:2]

        color_map = {
            "HEALTHY": (185, 245, 0),      # Неоновый изумрудно-зеленый
            "ROCKWOOL": (50, 220, 245),     # Золотисто-желтый цвет минваты (Rockwool Amber)
            "WARNING": (30, 150, 255),     # Насыщенный янтарно-оранжевый (Хлороз)
            "CRITICAL": (80, 60, 255),     # Ярко-алый (Плесень / Черная ножка)
            "EMPTY": (110, 100, 90)        # Приглушенный серо-синий
        }

        # Отрисовка рамки рабочей зоны матрицы (ROI)
        ymin, xmin, ymax, xmax = self.roi_norm
        rx1, ry1 = int(xmin * w), int(ymin * h)
        rx2, ry2 = int(xmax * w), int(ymax * h)
        cv2.rectangle(output, (rx1, ry1), (rx2, ry2), (0, 245, 185), 1, cv2.LINE_AA)

        for m in metrics:
            color = color_map.get(m.status, (150, 150, 150))
            cx, cy, r = m.center_x, m.center_y, m.radius

            # 1. Основное кольцо ячейки
            thickness = 2 if m.status in ("HEALTHY", "ROCKWOOL", "WARNING", "CRITICAL") else 1
            cv2.circle(output, (cx, cy), r, color, thickness, cv2.LINE_AA)

            # 2. Неоновое пульсирующее внешнее кольцо для предупреждений
            if m.status in ("WARNING", "CRITICAL"):
                cv2.circle(output, (cx, cy), r + 4, color, 1, cv2.LINE_AA)
            elif m.status == "ROCKWOOL":
                # Тонкий внутренний маркер для посадочной пробки 22 мм
                cv2.circle(output, (cx, cy), max(3, r - 5), color, 1, cv2.LINE_AA)

            # 3. Плашка с номером ячейки и процентом/статусом
            tag_text = f"{m.cell_id}"
            if m.status == "ROCKWOOL":
                tag_text += ": ВАТА"
            elif m.status != "EMPTY":
                tag_text += f": {int(m.coverage_ratio * 100)}%"

            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            tx = cx - tw // 2
            ty = cy + 4

            cv2.rectangle(output, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), (12, 16, 22), -1)
            cv2.rectangle(output, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), color, 1)
            cv2.putText(output, tag_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (250, 250, 250), 1, cv2.LINE_AA)

            # Маркер тревоги под ячейкой
            if m.status == "CRITICAL":
                alert_lbl = "! MOLD !" if m.stage == "MOLD" else "! SICK !"
                cv2.putText(output, alert_lbl, (cx - 20, cy + r + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (60, 60, 255), 1, cv2.LINE_AA)
            elif m.status == "WARNING":
                cv2.putText(output, "CHL", (cx - 10, cy + r + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (30, 150, 255), 1, cv2.LINE_AA)

        # 4. Общая статусная строка сверху кадра
        if show_summary:
            summary = self.get_summary(metrics)
            bar_h = 42
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 14, 20), -1)
            cv2.addWeighted(overlay, 0.82, output, 0.18, 0, output)

            cv2.line(output, (0, bar_h), (w, bar_h), (0, 245, 185), 1)

            title_str = f"AGROBOX 24-CELL MATRIX ({self.cols}x{self.rows})"
            cv2.putText(output, title_str, (14, 18), cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 245, 185), 1, cv2.LINE_AA)

            stat_str = (
                f"Ростки: {summary['active_cells']} | "
                f"Минвата: {summary.get('rockwool_cells', 0)} | "
                f"Здорово: {summary['healthy_cells']} | "
                f"Внимание: {summary['warning_cells']} | "
                f"Полог: {summary['avg_coverage_percent']}%"
            )
            cv2.putText(output, stat_str, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 230, 240), 1, cv2.LINE_AA)

        return output

    def to_json_dict(self, metrics: Optional[List[CellMetrics]] = None) -> Dict[str, Any]:
        """Сериализация данных для REST API веб-дашборда."""
        if metrics is None:
            metrics = self.last_metrics

        summary = self.get_summary(metrics)
        cells_data = []
        for m in metrics:
            cells_data.append({
                "id": m.cell_id,
                "index": m.index,
                "row": m.row,
                "col": m.col,
                "coverage": round(m.coverage_ratio * 100.0, 1),
                "chlorosis": round(m.chlorosis_ratio * 100.0, 1),
                "necrosis": round(m.necrosis_ratio * 100.0, 1),
                "mold": round(m.mold_ratio * 100.0, 1),
                "rockwool": round(m.rockwool_ratio * 100.0, 1),
                "exg": m.mean_exg,
                "health": m.health_index,
                "stage": m.stage,
                "status": m.status,
                "diag_ru": m.diagnosis_ru,
                "diag_en": m.diagnosis_en,
                "crop_name": m.crop_name,
                "planted_date": m.planted_date,
                "notes": m.notes,
                "is_manual_override": m.is_manual_override
            })

        return {
            "summary": summary,
            "cells": cells_data,
            "roi_norm": list(self.roi_norm),
            "rows": self.rows,
            "cols": self.cols,
            "cell_radius_ratio": self.cell_radius_ratio,
            "cell_overrides": self.cell_overrides,
            "timestamp": time.time()
        }
