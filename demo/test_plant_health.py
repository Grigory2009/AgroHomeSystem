#!/usr/bin/env python3
"""
AgroHomeSystem - Unit & Integration Tests for Plant Health Engine
Тестовый комплекс для проверки сегментации, классификаторов (ONNX, OpenCV DNN, TorchScript),
базы агрономических знаний и устойчивости алгоритма.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from plant_health_engine import (
    FoliageSegmenter,
    DiseaseClassifier,
    PlantHealthDetector,
    DiagnosisResult,
    LeafMetrics,
    AGRONOMIC_KNOWLEDGE_BASE
)


class TestPlantHealthEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.demo_dir = Path(__file__).parent.resolve()
        cls.test_image_path = cls.demo_dir / "test_leaf.jpg"
        if not cls.test_image_path.exists():
            raise FileNotFoundError(f"Test image not found at: {cls.test_image_path}")
        cls.test_img = cv2.imread(str(cls.test_image_path))
        assert cls.test_img is not None, "Failed to load test_leaf.jpg"

    def test_knowledge_base_integrity(self):
        """Verify that AGRONOMIC_KNOWLEDGE_BASE contains all 38 classes with complete schema."""
        self.assertEqual(len(AGRONOMIC_KNOWLEDGE_BASE), 38, "Knowledge base must have exactly 38 classes.")

        required_keys = {"crop", "crop_en", "disease_ru", "pathogen", "is_healthy", "severity", "treatment", "prevention"}
        healthy_count = 0
        diseased_count = 0

        for class_name, entry in AGRONOMIC_KNOWLEDGE_BASE.items():
            self.assertTrue(required_keys.issubset(entry.keys()), f"Missing keys in class: {class_name}")
            self.assertIsInstance(entry["is_healthy"], bool, f"is_healthy must be bool in {class_name}")
            self.assertTrue(len(entry["crop"]) > 0, f"Empty crop in {class_name}")
            self.assertTrue(len(entry["disease_ru"]) > 0, f"Empty disease_ru in {class_name}")
            self.assertTrue(len(entry["treatment"]) > 0, f"Empty treatment in {class_name}")

            if entry["is_healthy"]:
                healthy_count += 1
            else:
                diseased_count += 1

        self.assertEqual(healthy_count, 12, "There must be 12 healthy plant classes in PlantVillage.")
        self.assertEqual(diseased_count, 26, "There must be 26 disease classes in PlantVillage.")

    def test_foliage_segmenter(self):
        """Test FoliageSegmenter on synthetic image and real test leaf."""
        segmenter = FoliageSegmenter()

        # 1. Test on solid green image (synthetic perfect leaf)
        green_img = np.zeros((300, 300, 3), dtype=np.uint8)
        green_img[50:250, 50:250] = (30, 180, 50)  # BGR green foliage
        mask, metrics, boxes = segmenter.segment(green_img)

        self.assertEqual(mask.shape, (300, 300))
        self.assertGreater(metrics.total_leaf_pixels, 1000)
        self.assertGreater(metrics.healthy_green_ratio, 0.90)
        self.assertLess(metrics.chlorosis_ratio, 0.10)
        self.assertLess(metrics.necrosis_ratio, 0.10)
        self.assertGreater(len(boxes), 0)

        # 2. Test on real leaf
        mask_real, metrics_real, boxes_real = segmenter.segment(self.test_img)
        self.assertEqual(mask_real.shape, self.test_img.shape[:2])
        self.assertGreater(metrics_real.total_leaf_pixels, 500)
        self.assertTrue(0.0 <= metrics_real.healthy_green_ratio <= 1.0)
        self.assertTrue(0.0 <= metrics_real.chlorosis_ratio <= 1.0)
        self.assertTrue(0.0 <= metrics_real.necrosis_ratio <= 1.0)

    def test_classifier_onnx(self):
        """Test ONNX Runtime classifier backend."""
        try:
            clf = DiseaseClassifier(backend="onnx", num_threads=4)
            self.assertEqual(clf.engine_type, "onnx")
            label, conf, top_k, latency_ms = clf.predict(self.test_img)

            self.assertIn(label, AGRONOMIC_KNOWLEDGE_BASE)
            self.assertTrue(0.0 <= conf <= 1.0)
            self.assertEqual(len(top_k), 5)
            self.assertLess(latency_ms, 50.0, f"ONNX latency {latency_ms} ms is too high for CPU inference.")
        except Exception as e:
            self.fail(f"ONNX classifier failed: {e}")

    def test_classifier_opencv_dnn(self):
        """Test pure OpenCV DNN classifier backend."""
        try:
            clf = DiseaseClassifier(backend="opencv_dnn", num_threads=4)
            self.assertEqual(clf.engine_type, "opencv_dnn")
            label, conf, top_k, latency_ms = clf.predict(self.test_img)

            self.assertIn(label, AGRONOMIC_KNOWLEDGE_BASE)
            self.assertTrue(0.0 <= conf <= 1.0)
            self.assertEqual(len(top_k), 5)
            self.assertLess(latency_ms, 60.0, f"OpenCV DNN latency {latency_ms} ms is too high.")
        except Exception as e:
            self.fail(f"OpenCV DNN classifier failed: {e}")

    def test_classifier_torchscript(self):
        """Test PyTorch TorchScript classifier backend."""
        try:
            clf = DiseaseClassifier(backend="torchscript", num_threads=4)
            self.assertEqual(clf.engine_type, "torchscript")
            label, conf, top_k, latency_ms = clf.predict(self.test_img)

            self.assertIn(label, AGRONOMIC_KNOWLEDGE_BASE)
            self.assertTrue(0.0 <= conf <= 1.0)
            self.assertEqual(len(top_k), 5)
        except Exception as e:
            self.fail(f"TorchScript classifier failed: {e}")

    def test_detector_end_to_end(self):
        """Test complete PlantHealthDetector pipeline."""
        detector = PlantHealthDetector(backend="auto", num_threads=4)
        result: DiagnosisResult = detector.diagnose(self.test_img)

        # Check required output types and ranges
        self.assertIsInstance(result.crop_ru, str)
        self.assertIsInstance(result.disease_ru, str)
        self.assertTrue(0.0 <= result.confidence <= 100.0)
        self.assertTrue(0.0 <= result.health_index <= 100.0)
        self.assertIn(result.severity, ["None", "Mild", "Moderate", "Severe", "Critical", "Unknown"])
        self.assertTrue(len(result.treatment) > 0)
        self.assertTrue(len(result.prevention) > 0)

        # Check HUD drawing
        hud_frame = detector.draw_hud(self.test_img, result)
        self.assertEqual(hud_frame.shape, self.test_img.shape)

    def test_edge_case_black_image(self):
        """Test engine robustness on completely black image (simulating camera lens cap closed)."""
        detector = PlantHealthDetector(backend="auto")
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            result = detector.diagnose(black_img)
            self.assertIsNotNone(result)
            self.assertTrue(0.0 <= result.health_index <= 100.0)
        except Exception as e:
            self.fail(f"Detector crashed on black image: {e}")

    def test_edge_case_white_image(self):
        """Test engine robustness on overexposed white image."""
        detector = PlantHealthDetector(backend="auto")
        white_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        try:
            result = detector.diagnose(white_img)
            self.assertIsNotNone(result)
            self.assertTrue(0.0 <= result.health_index <= 100.0)
        except Exception as e:
            self.fail(f"Detector crashed on white image: {e}")


if __name__ == "__main__":
    unittest.main()
