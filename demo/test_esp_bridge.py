#!/usr/bin/env python3
"""
Unit tests for AgroHomeSystem ESP32-S3 Bridge and Telemetry Protocol.
"""

import unittest
import json
import sqlite3
import tempfile
import os
from pathlib import Path

from esp_bridge import determine_growth_stage, EspBridge, GROWTH_STAGES
from monitoring_dashboard import MonitoringSystem


class TestEspBridgeProtocol(unittest.TestCase):
    """Test suite for RPi <-> ESP32 protocol schemas and growth logic."""

    def test_determine_growth_stage_boundaries(self):
        """Verify growth percentage correctly maps to agronomic stages."""
        self.assertEqual(determine_growth_stage(0.0)["stage"], 1)
        self.assertEqual(determine_growth_stage(15.0)["stage"], 1)
        self.assertEqual(determine_growth_stage(25.0)["stage"], 1)
        self.assertEqual(determine_growth_stage(30.0)["stage"], 2)
        self.assertEqual(determine_growth_stage(55.0)["stage"], 2)
        self.assertEqual(determine_growth_stage(70.0)["stage"], 3)
        self.assertEqual(determine_growth_stage(85.0)["stage"], 3)
        self.assertEqual(determine_growth_stage(95.0)["stage"], 4)
        self.assertEqual(determine_growth_stage(100.0)["stage"], 4)
        # Clamping
        self.assertEqual(determine_growth_stage(-10.0)["stage"], 1)
        self.assertEqual(determine_growth_stage(150.0)["stage"], 4)

    def test_telemetry_database_schema(self):
        """Verify esp_telemetry table is created and stores sensor telemetry."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_db = tmp.name

        try:
            mon = MonitoringSystem(db_path=tmp_db, backend="opencv_dnn", enable_esp=False)
            
            # Simulate telemetry reception from ESP32
            test_telemetry = {
                "ph": 6.35,
                "tds": 845,
                "water_temp": 22.4,
                "air_temp": 24.1,
                "humidity": 56.5,
                "vpd": 1.12,
                "pump": True,
                "health_calc": 94
            }
            mon.on_esp_telemetry(test_telemetry)

            # Query database
            conn = sqlite3.connect(tmp_db)
            c = conn.cursor()
            c.execute("SELECT ph, tds, water_temp, air_temp, humidity, vpd, pump, health_calc FROM esp_telemetry")
            row = c.fetchone()
            conn.close()

            self.assertIsNotNone(row)
            self.assertAlmostEqual(row[0], 6.35, places=2)
            self.assertEqual(row[1], 845)
            self.assertAlmostEqual(row[2], 22.4, places=1)
            self.assertEqual(row[6], 1)
            self.assertEqual(row[7], 94)

        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)


if __name__ == "__main__":
    unittest.main()
