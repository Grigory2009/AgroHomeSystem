#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard with Statistics
Панель мониторинга с аналитикой и сохранением статистики
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from transformers import pipeline
from datetime import datetime, timedelta
import json
import sqlite3
from collections import defaultdict
import time
import sys

class MonitoringSystem:
    """Real-time monitoring with statistics and alerts"""
    
    def __init__(self, db_path="monitoring.db", model_name="AishaKanwal/ModelsViT_PlantDisease"):
        print("Инициализация системы мониторинга...")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db_path = db_path
        
        # Initialize database
        self.init_database()
        
        # Load models
        print("  Загрузка YOLO сегментации...")
        self.yolo = YOLO("yolov8n-seg.pt")
        
        print("  Загрузка классификатора болезней...")
        self.classifier = pipeline(
            "image-classification",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'disease_counts': defaultdict(int),
            'detections': [],
            'start_time': datetime.now()
        }
        
        print("[OK] Система готова к работе")
    
    def init_database(self):
        """Initialize SQLite database for logging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                diagnosis TEXT,
                confidence REAL,
                frame_number INTEGER,
                image_path TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                total_frames INTEGER,
                total_detections INTEGER,
                avg_confidence REAL,
                disease_distribution TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                disease TEXT,
                confidence REAL,
                alert_level TEXT,
                message TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_detection(self, diagnosis, confidence, frame_number, image_path=None):
        """Log detection to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO detections (timestamp, diagnosis, confidence, frame_number, image_path)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, diagnosis, confidence, frame_number, image_path))
        
        conn.commit()
        conn.close()
        
        # Update statistics
        self.stats['disease_counts'][diagnosis] += 1
        self.stats['detections'].append({
            'timestamp': timestamp,
            'diagnosis': diagnosis,
            'confidence': confidence
        })
        
        # Check for alerts
        self.check_alert(diagnosis, confidence)
    
    def check_alert(self, diagnosis, confidence):
        """Check if alert should be triggered"""
        if diagnosis == "Healthy":
            return
        
        # Determine alert level
        if confidence > 0.8:
            alert_level = "HIGH"
        elif confidence > 0.6:
            alert_level = "MEDIUM"
        else:
            alert_level = "LOW"
        
        # Log alert
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        message = f"Disease detected: {diagnosis} ({confidence*100:.1f}%)"
        
        cursor.execute("""
            INSERT INTO alerts (timestamp, disease, confidence, alert_level, message)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, diagnosis, confidence, alert_level, message))
        
        conn.commit()
        conn.close()
        
        # Console output
        if alert_level == "HIGH":
            print(f"\n[ALERT - HIGH] {message}")
        elif alert_level == "MEDIUM":
            print(f"[ALERT - MEDIUM] {message}")
    
    def process_frame(self, frame):
        """Process single frame and return results"""
        h, w = frame.shape[:2]
        results_list = []
        
        # YOLO segmentation
        results = self.yolo(frame, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None and len(result.masks) > 0:
                for idx, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
                    
                    # Get bounding box
                    box = result.boxes.xyxy[idx].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                    
                    if x2 - x1 > 10 and y2 - y1 > 10:
                        # Classify
                        mask_3d = np.stack([mask_np] * 3, axis=-1)
                        masked_region = (frame * mask_3d).astype(np.uint8)
                        
                        try:
                            cls_results = self.classifier(masked_region)
                            if cls_results:
                                results_list.append({
                                    'diagnosis': cls_results[0]['label'],
                                    'confidence': cls_results[0]['score'],
                                    'mask': mask_np,
                                    'box': (x1, y1, x2, y2)
                                })
                        except Exception as e:
                            print(f"Classification error: {e}")
        
        # Fallback
        if len(results_list) == 0:
            try:
                cls_results = self.classifier(frame)
                if cls_results:
                    results_list.append({
                        'diagnosis': cls_results[0]['label'],
                        'confidence': cls_results[0]['score'],
                        'mask': None,
                        'box': None
                    })
            except:
                pass
        
        return results_list
    
    def visualize_with_stats(self, frame, results_list):
        """Draw results and statistics on frame"""
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        disease_colors = {
            "Late_Blight": (0, 0, 255),
            "Early_Blight": (0, 165, 255),
            "Septoria_Leaf_Spot": (255, 0, 0),
            "Rust": (0, 165, 0),
            "Powdery_Mildew": (255, 255, 0),
            "Bacterial_Spot": (255, 0, 255),
            "Healthy": (0, 255, 0)
        }
        
        # Draw masks and boxes
        for result in results_list:
            diagnosis = result['diagnosis']
            confidence = result['confidence']
            mask = result['mask']
            box = result['box']
            
            color = disease_colors.get(diagnosis, (0, 255, 0))
            
            if mask is not None:
                mask_resized = cv2.resize(mask, (w, h))
                mask_uint8 = (mask_resized * 255).astype(np.uint8)
                
                overlay = display_frame.copy()
                overlay[mask_uint8 > 128] = cv2.addWeighted(
                    display_frame[mask_uint8 > 128], 0.6,
                    np.array(color, dtype=np.uint8), 0.4, 0
                )
                display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
            
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{diagnosis}: {confidence*100:.1f}%"
                cv2.putText(display_frame, label, (x1, max(20, y1 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw info panel
        panel_height = 120
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0)
        
        # Statistics
        uptime = datetime.now() - self.stats['start_time']
        uptime_str = str(uptime).split('.')[0]
        
        cv2.putText(display_frame, f"Frame: {self.stats['total_frames']}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Uptime: {uptime_str}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Detections: {len(self.stats['detections'])}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Disease distribution
        dist_str = ", ".join([f"{k}:{v}" for k, v in list(self.stats['disease_counts'].items())[:3]])
        cv2.putText(display_frame, f"Diseases: {dist_str}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Time
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(display_frame, time_str, (w - 150, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return display_frame
    
    def get_report(self):
        """Generate monitoring report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM detections")
        total_detections = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confidence) FROM detections")
        avg_confidence = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT diagnosis, COUNT(*) as count FROM detections GROUP BY diagnosis ORDER BY count DESC")
        disease_dist = cursor.fetchall()
        
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE alert_level = 'HIGH'")
        high_alerts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE alert_level = 'MEDIUM'")
        medium_alerts = cursor.fetchone()[0]
        
        conn.close()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'uptime': str(datetime.now() - self.stats['start_time']),
            'total_frames': self.stats['total_frames'],
            'total_detections': total_detections,
            'avg_confidence': float(avg_confidence),
            'disease_distribution': {disease: count for disease, count in disease_dist},
            'alerts': {
                'high': high_alerts,
                'medium': medium_alerts
            }
        }
        
        return report
    
    def save_report(self, filename="monitoring_report.json"):
        """Save report to file"""
        report = self.get_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        return filename
    
    def run(self, camera_id=0, save_dir=None, save_interval=300):
        """Run monitoring system"""
        print(f"\nЗапуск системы мониторинга...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"[ОШИБКА] Не удается открыть камеру {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print("[OK] Мониторинг запущен. Нажмите 'Q' для выхода.")
        
        last_save = time.time()
        import time as time_module
        frame_times = []
        
        try:
            while True:
                start_time = time_module.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.stats['total_frames'] += 1
                
                # Process frame
                results_list = self.process_frame(frame)
                
                # Log detections
                for result in results_list:
                    self.log_detection(
                        result['diagnosis'],
                        result['confidence'],
                        self.stats['total_frames']
                    )
                
                # Visualize
                display_frame = self.visualize_with_stats(frame, results_list)
                cv2.imshow("Real-time Monitoring", display_frame)
                
                # Calculate FPS
                frame_time = time_module.time() - start_time
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
                
                # Print status
                if self.stats['total_frames'] % 30 == 0:
                    report = self.get_report()
                    print(f"\n[Status] Frames: {report['total_frames']}, "
                          f"Detections: {report['total_detections']}, "
                          f"FPS: {fps:.1f}")
                
                # Save report periodically
                if time_module.time() - last_save > save_interval:
                    report_file = self.save_report()
                    print(f"[OK] Report saved: {report_file}")
                    last_save = time_module.time()
                
                # Save frame if detection
                if save_dir and len(results_list) > 0:
                    for result in results_list:
                        if result['diagnosis'] != 'Healthy':
                            filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                            cv2.imwrite(str(Path(save_dir) / filename), display_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nПрограмма прервана...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Save final report
            report_file = self.save_report("monitoring_report_final.json")
            print(f"[OK] Final report saved: {report_file}")
            
            # Print summary
            report = self.get_report()
            print("\n" + "="*60)
            print("MONITORING SUMMARY")
            print("="*60)
            print(f"Total Frames: {report['total_frames']}")
            print(f"Total Detections: {report['total_detections']}")
            print(f"Average Confidence: {report['avg_confidence']:.2%}")
            print(f"High Alerts: {report['alerts']['high']}")
            print(f"Medium Alerts: {report['alerts']['medium']}")
            print("\nDisease Distribution:")
            for disease, count in report['disease_distribution'].items():
                print(f"  {disease}: {count}")
            print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Monitoring with Statistics")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory for detected images")
    parser.add_argument("--save-interval", type=int, default=300, help="Save report every N seconds")
    parser.add_argument("--db", type=str, default="monitoring.db", help="Database file")
    args = parser.parse_args()
    
    # Run monitoring
    monitor = MonitoringSystem(db_path=args.db)
    monitor.run(camera_id=args.camera, save_dir=args.save_dir, save_interval=args.save_interval)


if __name__ == "__main__":
    main()
