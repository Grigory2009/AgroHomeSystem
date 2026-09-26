"""
AgroHomeSystem - Per-Cell Plant Tracking Engine (24-Cell Matrix)
=============================================================================
Модуль индивидуального поячеечного трекинга для 3D-корпуса AgroHomeSystem v2
(лоток на 24 ячейки: сетка 4x6 под минераловатные пробки 22 мм).

Возможности:
1. Автоматическая разбивка рабочей зоны лотка на 24 независимых микро-ROI (4x6).
2. Расчет биофизических параметров для каждой ячейки:
   - Процент покрытия пологом (Biomass Coverage %)
   - Индекс хлорофилла (ExG)
   - Доля хлороза (Chlorosis %) и некроза (Necrosis %)
   - Детекция мицелия серой гнили (Mold / Botrytis %)
3. Определение фазы и статуса для каждой ячейки:
   - EMPTY (пустая/нет всходов)
   - GERMINATING (фаза прорастания семян)
   - SPROUT (семядоли / молодой росток)
   - HEALTHY (активный здоровый рост)
   - CHLOROSIS (дефицит питания/pH стресс)
   - MOLD (плесень/серая гниль)
   - DAMPING_OFF (черная ножка)
4. Рендеринг киберпанк HUD-оверлея (неоновые кольца, бейджи, статусы).
5. Экспорт состояния для Web Dashboard (JSON API / WebSocket).
=============================================================================
"""

import time
import math
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
    chlorosis_pixels: int = 0         # Пикселей пожелтения/хлороза
    necrosis_pixels: int = 0          # Пикселей отмершей ткани
    mold_pixels: int = 0              # Пикселей мицелия плесени
    
    coverage_ratio: float = 0.0       # 0.0 .. 1.0 (площадь покрытия ячейки)
    chlorosis_ratio: float = 0.0      # Доля хлороза от биомассы
    necrosis_ratio: float = 0.0       # Доля некроза от биомассы
    mold_ratio: float = 0.0           # Доля мицелия от ячейки
    mean_exg: float = 0.0             # Средний индекс Excess Green
    
    health_index: float = 100.0       # 0.0 .. 100.0 %
    stage: str = "EMPTY"              # "EMPTY", "GERMINATING", "SPROUT", "MATURE"
    status: str = "EMPTY"             # "HEALTHY", "WARNING", "CRITICAL", "EMPTY"
    diagnosis_ru: str = "Свободно"
    diagnosis_en: str = "Empty"
    
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
        cell_radius_ratio: float = 0.38
    ):
        """
        :param rows: Количество рядов (по умолчанию 6)
        :param cols: Количество колонок (по умолчанию 4)
        :param roi_norm: Относительные границы лотка в кадре (ymin, xmin, ymax, xmax) в диапазоне 0.0..1.0
        :param cell_radius_ratio: Радиус окружности ячейки относительно шага сетки
        """
        self.rows = rows
        self.cols = cols
        self.roi_norm = roi_norm
        self.cell_radius_ratio = cell_radius_ratio
        
        # История динамики прироста биомассы по ячейкам (для расчета скорости роста)
        self.history: Dict[str, List[Tuple[float, float]]] = {}  # cell_id -> [(timestamp, coverage)]
        self.last_metrics: List[CellMetrics] = []

    def set_roi(self, ymin: float, xmin: float, ymax: float, xmax: float):
        """Калибровка границ лотка в поле зрения камеры."""
        self.roi_norm = (
            max(0.0, min(ymin, 0.8)),
            max(0.0, min(xmin, 0.8)),
            min(1.0, max(ymax, 0.2)),
            min(1.0, max(xmax, 0.2))
        )

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

        row_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
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
        Комплексный анализ всех 24 ячеек на кадре.
        :param frame_bgr: Исходный кадр с камеры (BGR uint8)
        :param exg_img: Предрассчитанный индекс Excess Green (опционально)
        :param foliage_mask: Маска листвы (опционально, иначе вычисляется автоматически)
        :return: Список метрик для каждой из 24 ячеек
        """
        h, w = frame_bgr.shape[:2]
        now = time.time()

        # 1. Быстрый расчет спектральных масок, если они не переданы
        if exg_img is None:
            b = frame_bgr[:, :, 0].astype(np.float32)
            g = frame_bgr[:, :, 1].astype(np.float32)
            r = frame_bgr[:, :, 2].astype(np.float32)
            exg_raw = 2.0 * g - r - b
            exg_img = np.clip(exg_raw, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Зеленая активная листва
        lower_green = np.array([25, 25, 25], dtype=np.uint8)
        upper_green = np.array([95, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_active = np.where((exg_img > 15) & (green_mask > 0), 255, 0).astype(np.uint8)

        # Хлороз (желтоватые оттенки)
        lower_yellow = np.array([16, 25, 50], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        chlorosis_mask = np.where((yellow_mask > 0) & (exg_img > 0), 255, 0).astype(np.uint8)

        # Листва (зелень + хлороз)
        if foliage_mask is None:
            foliage_mask = cv2.bitwise_or(green_active, chlorosis_mask)

        # Некроз (темно-коричневые/черные пятна)
        lower_necrosis = np.array([5, 40, 20], dtype=np.uint8)
        upper_necrosis = np.array([25, 255, 90], dtype=np.uint8)
        necrosis_mask = cv2.inRange(hsv, lower_necrosis, upper_necrosis)

        # Мицелий серой гнили (Botrytis / плесень: низкая насыщенность, средняя/высокая яркость)
        lower_mold = np.array([0, 0, 110], dtype=np.uint8)
        upper_mold = np.array([180, 45, 210], dtype=np.uint8)
        mold_candidate = cv2.inRange(hsv, lower_mold, upper_mold)
        # Исключаем чистый субстрат вне посадочной зоны
        mold_mask = np.where((mold_candidate > 0) & (exg_img < 15), 255, 0).astype(np.uint8)

        grid_geo = self._generate_grid_geometry(w, h)
        results: List[CellMetrics] = []

        for item in grid_geo:
            cid = item["cell_id"]
            cx = item["cx"]
            cy = item["cy"]
            rad = item["radius"]
            x1, y1, x2, y2 = item["bbox"]

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

            coverage = round(foliage_px / float(total_px), 3)
            chl_ratio = round(chl_px / float(max(1, foliage_px)), 3)
            nec_ratio = round(nec_px / float(max(1, foliage_px)), 3)
            mld_ratio = round(mld_px / float(total_px), 3)

            # Средний ExG внутри ячейки
            cell_exg_vals = exg_img[cell_circle_mask > 0]
            mean_exg = float(np.mean(cell_exg_vals)) if len(cell_exg_vals) > 0 else 0.0

            # Определение стадии и здоровья ячейки
            stage = "EMPTY"
            status = "EMPTY"
            diag_ru = "Свободно"
            diag_en = "Empty"
            health_idx = 100.0

            if coverage < 0.04:
                stage = "EMPTY"
                status = "EMPTY"
                diag_ru = "Свободно (нет всходов)"
                diag_en = "Empty / No Sprouts"
                health_idx = 100.0
            elif mld_ratio >= 0.06:
                stage = "MOLD"
                status = "CRITICAL"
                diag_ru = "Плесень (Серая гниль)"
                diag_en = "Gray Mold (Botrytis)"
                health_idx = max(5.0, 30.0 - mld_ratio * 100.0)
            elif nec_ratio >= 0.15 and coverage > 0.08:
                stage = "DAMPING_OFF"
                status = "CRITICAL"
                diag_ru = "Полегание (Черная ножка)"
                diag_en = "Damping-Off (Pythium)"
                health_idx = max(10.0, 40.0 - nec_ratio * 100.0)
            elif chl_ratio >= 0.20:
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
                health_idx = 95.0
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
                coverage_ratio=coverage,
                chlorosis_ratio=chl_ratio,
                necrosis_ratio=nec_ratio,
                mold_ratio=mld_ratio,
                mean_exg=round(mean_exg, 1),
                health_index=round(health_idx, 1),
                stage=stage,
                status=status,
                diagnosis_ru=diag_ru,
                diagnosis_en=diag_en,
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
        healthy = sum(1 for m in metrics if m.status == "HEALTHY")
        warning = sum(1 for m in metrics if m.status == "WARNING")
        critical = sum(1 for m in metrics if m.status == "CRITICAL")
        active = total - empty

        avg_cov = float(np.mean([m.coverage_ratio for m in metrics])) if metrics else 0.0
        avg_health = float(np.mean([m.health_index for m in metrics if m.status != "EMPTY"])) if active > 0 else 100.0

        return {
            "total_cells": total,
            "active_cells": active,
            "empty_cells": empty,
            "healthy_cells": healthy,
            "warning_cells": warning,
            "critical_cells": critical,
            "avg_coverage_percent": round(avg_cov * 100.0, 1),
            "avg_health_score": round(avg_health, 1),
            "layout": f"{self.cols}x{self.rows}"
        }

    def draw_hud(
        self,
        frame: np.ndarray,
        metrics: Optional[List[CellMetrics]] = None,
        show_summary: bool = True
    ) -> np.ndarray:
        """
        Отрисовка технологичного киберпанк HUD-оверлея на кадре:
        - Неоновые кольца вокруг каждой из 24 ячеек
        - Цветовая дифференциация: Изумрудный (норма), Янтарный (хлороз), Алый (болезнь/плесень)
        - Бейдж с ID ячейки и процентом покрытия
        - Общая статус-панель сверху
        """
        if metrics is None:
            metrics = self.last_metrics

        output = frame.copy()
        h, w = output.shape[:2]

        # Цвета статусов в формате BGR
        color_map = {
            "HEALTHY": (185, 245, 0),      # Неоновый изумрудно-зеленый #00F5B9
            "WARNING": (50, 180, 255),     # Янтарно-оранжевый
            "CRITICAL": (80, 60, 255),     # Ярко-алый
            "EMPTY": (110, 100, 90)        # Приглушенный серо-синий
        }

        for m in metrics:
            color = color_map.get(m.status, (150, 150, 150))
            cx, cy, r = m.center_x, m.center_y, m.radius

            # 1. Основное кольцо ячейки
            thickness = 2 if m.status in ("HEALTHY", "WARNING", "CRITICAL") else 1
            cv2.circle(output, (cx, cy), r, color, thickness, cv2.LINE_AA)

            # 2. Неоновое пульсирующее внешнее кольцо для предупреждений
            if m.status in ("WARNING", "CRITICAL"):
                cv2.circle(output, (cx, cy), r + 4, color, 1, cv2.LINE_AA)

            # 3. Плашка с номером ячейки и процентом
            tag_text = f"{m.cell_id}"
            if m.status != "EMPTY":
                tag_text += f": {int(m.coverage_ratio * 100)}%"

            # Подложка под текст
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            tx = cx - tw // 2
            ty = cy + 4

            cv2.rectangle(output, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), (12, 16, 22), -1)
            cv2.rectangle(output, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), color, 1)
            cv2.putText(output, tag_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (250, 250, 250), 1, cv2.LINE_AA)

            # Маркер тревоги под ячейкой
            if m.status == "CRITICAL":
                alert_lbl = "! MOLD !" if m.stage == "MOLD" else "! SICK !"
                cv2.putText(output, alert_lbl, (cx - 20, cy + r + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (60, 60, 255), 1, cv2.LINE_AA)
            elif m.status == "WARNING":
                cv2.putText(output, "CHL", (cx - 10, cy + r + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (50, 180, 255), 1, cv2.LINE_AA)

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
                f"Active: {summary['active_cells']}/24 | "
                f"Healthy: {summary['healthy_cells']} | "
                f"Warn: {summary['warning_cells']} | "
                f"Crit: {summary['critical_cells']} | "
                f"Canopy: {summary['avg_coverage_percent']}%"
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
                "exg": m.mean_exg,
                "health": m.health_index,
                "stage": m.stage,
                "status": m.status,
                "diag_ru": m.diagnosis_ru,
                "diag_en": m.diagnosis_en
            })

        return {
            "summary": summary,
            "cells": cells_data,
            "timestamp": time.time()
        }
