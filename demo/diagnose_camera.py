#!/usr/bin/env python3
"""
AgroHomeSystem - Real-time Camera Plant Health Diagnostics
Диагностика здоровья растений в реальном времени с веб-камеры / камеры Raspberry Pi.
Поддерживает автоматический поиск и выбор камер, горячее переключение на лету и экспорт отчетов.
"""

import sys
import os

# Suppress noisy OpenCV backend probe warnings
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import time
import json
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import cv2
try:
    cv2.setLogLevel(0)
except Exception:
    pass
import numpy as np

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from plant_health_engine import PlantHealthDetector, DiagnosisResult


def get_camera_backend() -> int:
    """Return platform-specific preferred OpenCV camera backend."""
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return 0


def scan_available_cameras(max_tested: int = 6) -> List[Dict[str, Any]]:
    """Scan system for all active and accessible camera devices."""
    available: List[Dict[str, Any]] = []
    preferred_backend = get_camera_backend()

    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, preferred_backend) if preferred_backend != 0 else cv2.VideoCapture(idx)
        if not cap.isOpened() and preferred_backend != 0:
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            # Attempt to read frame with short timeout
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                available.append({
                    "id": idx,
                    "width": w,
                    "height": h,
                    "description": f"Камера {idx} ({w}x{h})"
                })
            cap.release()

    return available


def select_camera_interactively(cameras: List[Dict[str, Any]]) -> int:
    """Prompt user to choose a camera or auto-select best camera."""
    if not cameras:
        print("[ВНИМАНИЕ] Камеры не обнаружены автоматически. Попытка открыть камеру 0 по умолчанию.")
        return 0

    if len(cameras) == 1:
        chosen = cameras[0]["id"]
        print(f"[OK] Найдена 1 камера: {cameras[0]['description']}. Запуск...")
        return chosen

    # Recommend camera with largest resolution (e.g. 1280x720 > 640x480)
    best_cam = max(cameras, key=lambda c: c["width"] * c["height"])
    recommended_id = best_cam["id"]

    print("\n" + "=" * 60)
    print("      📷 ВЫБОР КАМЕРЫ ДЛЯ ДИАГНОСТИКИ РАСТЕНИЙ 📷")
    print("=" * 60)
    print("Обнаружены следующие камеры:")
    for cam in cameras:
        rec_tag = " <-- [Рекомендуется: Основная HD]" if cam["id"] == recommended_id else ""
        print(f"  [{cam['id']}] {cam['description']}{rec_tag}")
    print("-" * 60)

    try:
        user_input = input(f"Введите номер камеры [по умолчанию {recommended_id}]: ").strip()
        if user_input.isdigit() and int(user_input) in [c["id"] for c in cameras]:
            return int(user_input)
    except Exception:
        pass

    print(f"Выбрана камера {recommended_id} по умолчанию.")
    return recommended_id


class VideoStream:
    """
    Robust threaded video stream with auto-reconnect and DirectShow/V4L2 support.
    """

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720):
        self.src = src
        self.width = width
        self.height = height
        self.preferred_backend = get_camera_backend()

        self.stream = self._open_capture(src)
        self.grabbed, self.frame = self._initial_warmup()

        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _open_capture(self, src: int) -> cv2.VideoCapture:
        """Open capture device with platform-specific backend."""
        cap = cv2.VideoCapture(src, self.preferred_backend) if self.preferred_backend != 0 else cv2.VideoCapture(src)
        if not cap.isOpened() and self.preferred_backend != 0:
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру с индексом {src}.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _initial_warmup(self, max_attempts: int = 10) -> Tuple[bool, Optional[np.ndarray]]:
        """Warm up camera sensor with retry loop."""
        for _ in range(max_attempts):
            ret, frame = self.stream.read()
            if ret and frame is not None and frame.size > 0:
                return True, frame
            time.sleep(0.1)
        return False, None

    def update(self) -> None:
        consecutive_failures = 0
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed and frame is not None and frame.size > 0:
                consecutive_failures = 0
                with self.lock:
                    self.grabbed = True
                    self.frame = frame
            else:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    # After 30 failed frames (~1.5s), mark stopped
                    self.stopped = True
                    break
                time.sleep(0.05)

            time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self) -> None:
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.stream.release()


class RealtimePlantMonitor:
    """Interactive Realtime Plant Monitor with HUD and hotkeys."""

    def __init__(
        self,
        camera_id: int = 0,
        backend: str = "auto",
        num_threads: int = 4,
        inference_fps: float = 6.0,
        width: int = 1280,
        height: int = 720,
        save_dir: str = "snapshots",
        available_cameras: Optional[List[int]] = None
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.backend = backend
        self.num_threads = num_threads
        self.inference_interval = 1.0 / max(0.5, inference_fps)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.available_cameras = available_cameras or [camera_id]

        self.detector = PlantHealthDetector(backend=backend, num_threads=num_threads)

        self.last_inference_time = 0.0
        self.last_result: Optional[DiagnosisResult] = None
        self.smoothed_health_index: float = 100.0

        self.show_mask = True
        self.show_treatment_help = False
        self.paused = False

    def smooth_prediction(self, result: DiagnosisResult) -> None:
        alpha = 0.35
        self.smoothed_health_index = (alpha * result.health_index) + ((1.0 - alpha) * self.smoothed_health_index)
        result.health_index = round(self.smoothed_health_index, 1)

    def draw_treatment_overlay(self, frame: np.ndarray, result: DiagnosisResult) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        panel_y1 = max(90, h - 230)
        cv2.rectangle(overlay, (10, panel_y1), (w - 10, h - 35), (20, 25, 30), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (10, panel_y1), (w - 10, h - 35), (0, 200, 255), 1)

        cv2.putText(frame, "РЕКОМЕНДАЦИИ ПО ЛЕЧЕНИЮ И МИКРОКЛИМАТУ:", (25, panel_y1 + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)

        pathogen = f"Возбудитель: {result.pathogen} | Тяжесть: {result.severity}"
        y = panel_y1 + 55
        for line_title, line_text in [("Терапия: ", result.treatment), ("Профилактика: ", result.prevention), ("Патоген: ", pathogen)]:
            full_txt = line_title + line_text
            if len(full_txt) > 85:
                part1 = full_txt[:82] + "..."
                cv2.putText(frame, part1, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, full_txt, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            y += 26

        cv2.putText(frame, "[Нажмите 'H' чтобы скрыть рекомендации]", (25, panel_y1 + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 180, 190), 1, cv2.LINE_AA)
        return frame

    def draw_footer_hotkeys(self, frame: np.ndarray, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 32), (w, h), (10, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cam_text = f"Камера: [{self.camera_id}] (C: сменить)" if len(self.available_cameras) > 1 else f"Камера: [{self.camera_id}]"
        hotkey_str = f"Q: Выход | S: Снимок | P: Пауза | H: Лечение | M: Маска | {cam_text}"
        cv2.putText(frame, hotkey_str, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 210, 220), 1, cv2.LINE_AA)

        fps_str = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_str, (w - 110, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 180), 1, cv2.LINE_AA)
        return frame

    def save_snapshot(self, frame: np.ndarray, result: DiagnosisResult) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"plant_{timestamp}.jpg"
        json_name = f"plant_{timestamp}.json"

        img_path = self.save_dir / img_name
        json_path = self.save_dir / json_name

        cv2.imwrite(str(img_path), frame)

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "camera_id": self.camera_id,
            "crop": result.crop_ru,
            "crop_en": result.crop_en,
            "diagnosis": result.disease_ru,
            "confidence_percent": result.confidence,
            "health_index": result.health_index,
            "is_healthy": result.is_healthy,
            "severity": result.severity,
            "pathogen": result.pathogen,
            "treatment": result.treatment,
            "prevention": result.prevention,
            "biophysical_metrics": {
                "healthy_green_ratio": result.metrics.healthy_green_ratio,
                "chlorosis_ratio": result.metrics.chlorosis_ratio,
                "necrosis_ratio": result.metrics.necrosis_ratio,
                "mean_exg": result.metrics.mean_exg,
                "segments_found": result.metrics.contour_count
            },
            "image_file": img_name
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"[OK] Снимок и отчет сохранены: {img_path}")
        return img_path

    def run(self) -> None:
        window_name = "AgroHomeSystem - Plant Health Live"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        switch_requested = False
        next_cam_id = self.camera_id

        while True:
            print(f"\nЗапуск видеопотока с камеры {self.camera_id}...")
            try:
                vs = VideoStream(src=self.camera_id, width=self.width, height=self.height)
            except Exception as e:
                print(f"[ОШИБКА] Не удалось открыть камеру {self.camera_id}: {e}")
                if len(self.available_cameras) > 1:
                    # Switch to next camera
                    cur_idx = self.available_cameras.index(self.camera_id) if self.camera_id in self.available_cameras else 0
                    self.camera_id = self.available_cameras[(cur_idx + 1) % len(self.available_cameras)]
                    print(f"Попытка переключиться на камеру {self.camera_id}...")
                    continue
                else:
                    break

            fps_history: List[float] = []
            switch_requested = False

            try:
                while not switch_requested:
                    t_frame_start = time.perf_counter()

                    if not self.paused:
                        grabbed, raw_frame = vs.read()
                        if not grabbed or raw_frame is None:
                            time.sleep(0.02)
                            continue

                        now = time.perf_counter()
                        if (now - self.last_inference_time) >= self.inference_interval or self.last_result is None:
                            self.last_result = self.detector.diagnose(raw_frame)
                            self.smooth_prediction(self.last_result)
                            self.last_inference_time = now

                        display_frame = self.detector.draw_hud(raw_frame, self.last_result, show_mask=self.show_mask)

                        if self.show_treatment_help and self.last_result:
                            display_frame = self.draw_treatment_overlay(display_frame, self.last_result)

                        t_frame_end = time.perf_counter()
                        frame_dt = t_frame_end - t_frame_start
                        current_fps = 1.0 / max(frame_dt, 0.001)
                        fps_history.append(current_fps)
                        if len(fps_history) > 30:
                            fps_history.pop(0)
                        avg_fps = sum(fps_history) / len(fps_history)

                        display_frame = self.draw_footer_hotkeys(display_frame, avg_fps)
                        cv2.imshow(window_name, display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        vs.stop()
                        cv2.destroyAllWindows()
                        print("\nМониторинг завершен.")
                        return
                    elif key in (ord('s'), ord('S')):
                        if self.last_result and 'display_frame' in locals():
                            self.save_snapshot(display_frame, self.last_result)
                    elif key in (ord('p'), ord('P')):
                        self.paused = not self.paused
                    elif key in (ord('h'), ord('H')):
                        self.show_treatment_help = not self.show_treatment_help
                    elif key in (ord('m'), ord('M')):
                        self.show_mask = not self.show_mask
                    elif key in (ord('c'), ord('C')) and len(self.available_cameras) > 1:
                        # Switch camera hotkey
                        cur_idx = self.available_cameras.index(self.camera_id) if self.camera_id in self.available_cameras else 0
                        self.camera_id = self.available_cameras[(cur_idx + 1) % len(self.available_cameras)]
                        print(f"\n[ПЕРЕКЛЮЧЕНИЕ] Переход на камеру {self.camera_id}...")
                        switch_requested = True
                        break

            finally:
                vs.stop()

            if not switch_requested:
                break

        cv2.destroyAllWindows()
        print("\nМониторинг завершен.")


def main():
    parser = argparse.ArgumentParser(description="AgroHomeSystem - Real-time Camera Plant Diagnostics")
    parser.add_argument("--camera", "-c", type=int, default=None, help="Camera ID (default: auto-detect or interactive)")
    parser.add_argument("--auto", action="store_true", help="Auto-select recommended camera without prompt")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras and exit")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    parser.add_argument("--backend", "-b", type=str, default="auto", choices=["auto", "onnx", "opencv_dnn", "torchscript", "transformers"])
    parser.add_argument("--threads", "-t", type=int, default=4, help="CPU threads (default: 4 for RPi 4)")
    parser.add_argument("--inference-rate", "-r", type=float, default=6.0, help="Inference cadence in Hz (default: 6.0 Hz)")
    parser.add_argument("--save-dir", type=str, default="snapshots", help="Directory for snapshots")
    args = parser.parse_args()

    # Scan available cameras
    detected_cameras = scan_available_cameras()

    if args.list_cameras:
        print("Обнаруженные камеры в системе:")
        for cam in detected_cameras:
            print(f"  • ID {cam['id']}: {cam['description']}")
        return

    # Determine camera ID
    if args.camera is not None:
        camera_id = args.camera
    elif args.auto:
        best_cam = max(detected_cameras, key=lambda c: c["width"] * c["height"]) if detected_cameras else {"id": 0}
        camera_id = best_cam["id"]
        print(f"[АВТО] Выбрана камера {camera_id} ({best_cam.get('description', '')})")
    else:
        camera_id = select_camera_interactively(detected_cameras)

    all_ids = [c["id"] for c in detected_cameras] if detected_cameras else [camera_id]
    if camera_id not in all_ids:
        all_ids.append(camera_id)

    monitor = RealtimePlantMonitor(
        camera_id=camera_id,
        backend=args.backend,
        num_threads=args.threads,
        inference_fps=args.inference_rate,
        width=args.width,
        height=args.height,
        save_dir=args.save_dir,
        available_cameras=all_ids
    )
    monitor.run()


if __name__ == "__main__":
    main()
