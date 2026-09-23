#!/usr/bin/env python3
"""
AgroHomeSystem - Web Interface for Plant Health & Disease Diagnostics
Streamlit UI для визуальной диагностики здоровья растений с камеры и по фото.
Оптимизировано для Raspberry Pi 4 (8GB) и персональных компьютеров.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
from PIL import Image

# pyrefly: ignore [missing-import]
import streamlit as st
import cv2
import numpy as np

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from plant_health_engine import PlantHealthDetector, DiagnosisResult, AGRONOMIC_KNOWLEDGE_BASE

st.set_page_config(
    page_title="AgroHomeSystem - Plant Health",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern agronomic dashboard
st.markdown("""
    <style>
    .main { padding: 1.5rem; }
    .stButton button {
        width: 100%;
        padding: 0.6rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.6rem;
        background-color: #f7f9fa;
        border-left: 5px solid #2e7d32;
        margin-bottom: 0.8rem;
    }
    .alert-healthy {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #1b5e20;
    }
    .alert-warning {
        background-color: #fff8e1;
        border-left: 5px solid #f57f17;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #e65100;
    }
    .alert-danger {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #b71c1c;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Load optimized plant health detector once."""
    return PlantHealthDetector(backend="auto", num_threads=4)


def main():
    detector = load_detector()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🌱 AgroHomeSystem: Plant Health AI")
        st.markdown(f"**Edge-оптимизированная диагностика здоровья растений (RPi 4)** | Активный движок: `{detector.classifier.engine_type.upper()}`")
    with col2:
        st.metric("Статус системы", "Готов ✓", delta=f"{detector.classifier.engine_type.upper()}")

    # Sidebar navigation
    with st.sidebar:
        st.header("⚙️ Режим работы")
        mode = st.radio("Выберите режим", ["📷 Веб-камера (Real-time)", "📤 Загрузка фото", "📚 Справочник болезней (38)"])
        st.markdown("---")
        st.subheader("Характеристики системы")
        st.info(f"""
        - **Архитектура:** MobileNetV2 + ExG Segmentation
        - **Классов:** 38 патологий и здоровых культур
        - **Оптимизация:** Raspberry Pi 4 (8GB)
        - **Движок:** {detector.classifier.engine_type.upper()}
        - **Память:** ~60 МБ RAM
        """)

    # 1. Real-time Camera
    if mode == "📷 Веб-камера (Real-time)":
        st.subheader("Режим реального времени")

        c1, c2 = st.columns([2, 1])
        with c1:
            video_placeholder = st.empty()
        with c2:
            diag_placeholder = st.empty()
            metrics_placeholder = st.empty()
            treat_placeholder = st.empty()

        # Camera discovery
        from diagnose_camera import scan_available_cameras, get_camera_backend
        cams = scan_available_cameras()
        cam_options = {}
        default_index = 0
        if cams:
            best_id = max(cams, key=lambda c: c["width"] * c["height"])["id"]
            for i, c in enumerate(cams):
                tag = " (Рекомендуется)" if c["id"] == best_id else ""
                label = f"{c['description']}{tag}"
                cam_options[label] = c["id"]
                if c["id"] == best_id:
                    default_index = i
        else:
            cam_options["Камера 0 (По умолчанию)"] = 0

        col_cam, col_fps, col_rate = st.columns(3)
        with col_cam:
            chosen_cam_label = st.selectbox("Камера", list(cam_options.keys()), index=default_index)
            camera_id = cam_options[chosen_cam_label]
        with col_fps:
            target_fps = st.slider("Целевой FPS рендера", 5, 30, 15)
        with col_rate:
            infer_rate = st.slider("Частота инференса нейросети (Hz)", 1, 15, 5)

        run_camera = st.checkbox("Запустить видеопоток", value=False)

        if run_camera:
            preferred_backend = get_camera_backend()
            cap = cv2.VideoCapture(int(camera_id), preferred_backend) if preferred_backend != 0 else cv2.VideoCapture(int(camera_id))
            if not cap.isOpened() and preferred_backend != 0:
                cap = cv2.VideoCapture(int(camera_id))

            if not cap.isOpened():
                st.error(f"Не удалось открыть камеру {camera_id}.")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            last_infer_time = 0.0
            last_diag = None
            infer_interval = 1.0 / infer_rate
            frame_count = 0
            t_start = time.time()

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Нет сигнала с камеры.")
                    break

                now = time.time()
                if (now - last_infer_time) >= infer_interval or last_diag is None:
                    last_diag = detector.diagnose(frame)
                    last_infer_time = now

                display_frame = detector.draw_hud(frame, last_diag)
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

                # Render diagnostic cards in sidebar/col2
                with diag_placeholder.container():
                    hi = last_diag.health_index
                    if hi >= 80:
                        st.markdown(f"""
                        <div class="alert-healthy">
                        <h3>✓ Растение здорово</h3>
                        <b>Культура:</b> {last_diag.crop_ru}<br>
                        <b>Индекс здоровья:</b> {hi:.1f}%<br>
                        <b>Уверенность:</b> {last_diag.confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    elif hi >= 50:
                        st.markdown(f"""
                        <div class="alert-warning">
                        <h3>⚠️ Внимание: {last_diag.disease_ru}</h3>
                        <b>Культура:</b> {last_diag.crop_ru}<br>
                        <b>Индекс здоровья:</b> {hi:.1f}%<br>
                        <b>Тяжесть:</b> {last_diag.severity}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-danger">
                        <h3>🚨 Тревога: {last_diag.disease_ru}</h3>
                        <b>Культура:</b> {last_diag.crop_ru}<br>
                        <b>Индекс здоровья:</b> {hi:.1f}%<br>
                        <b>Возбудитель:</b> {last_diag.pathogen}
                        </div>
                        """, unsafe_allow_html=True)

                with metrics_placeholder.container():
                    m = last_diag.metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Зелень", f"{m.healthy_green_ratio*100:.0f}%")
                    m2.metric("Хлороз", f"{m.chlorosis_ratio*100:.0f}%")
                    m3.metric("Некроз", f"{m.necrosis_ratio*100:.0f}%")

                with treat_placeholder.container():
                    st.markdown("**Терапия:** " + last_diag.treatment)

                frame_count += 1
                time.sleep(1.0 / target_fps)

            cap.release()

    # 2. Upload Image
    elif mode == "📤 Загрузка фото":
        st.subheader("Анализ одиночной фотографии листа / растения")
        uploaded_file = st.file_uploader("Выберите фото растения (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png", "bmp", "webp"])

        # Quick test samples
        st.markdown("Или выберите тестовый образец из каталога:")
        test_samples = ["test_leaf.jpg", "sample_pepper_healthy.jpg", "sample_corn_rust.jpg", "sample_peach_healthy.jpg", "sample_pepper_bacterial_spot.jpg"]
        chosen_sample = st.selectbox("Тестовые изображения", ["-- Выберите --"] + test_samples)

        frame = None
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif chosen_sample != "-- Выберите --" and Path(chosen_sample).exists():
            frame = cv2.imread(chosen_sample)

        if frame is not None:
            with st.spinner("Диагностика образца..."):
                result: DiagnosisResult = detector.diagnose(frame)
                annotated = detector.draw_hud(frame, result)

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("### Анализ изображения с HUD")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

            with col2:
                st.markdown("### Заключение агрономической экспертизы")
                hi = result.health_index
                if result.is_healthy:
                    st.success(f"✓ {result.crop_ru}: Здоровое растение ({result.confidence:.1f}%)")
                else:
                    st.error(f"⚠️ {result.crop_ru}: {result.disease_ru} ({result.confidence:.1f}%)")

                st.progress(int(hi))
                st.write(f"**Интегральный индекс жизнеспособности:** {hi:.1f} / 100")

                st.markdown(f"**Возбудитель:** {result.pathogen}")
                st.markdown(f"**Тяжесть заболевания:** {result.severity}")
                st.markdown(f"**Время инференса:** {result.processing_time_ms:.1f} мс")

                st.markdown("---")
                st.markdown("#### Биофизические индексы")
                m = result.metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Здоровая ткань", f"{m.healthy_green_ratio*100:.1f}%")
                m2.metric("Хлороз (желтизна)", f"{m.chlorosis_ratio*100:.1f}%")
                m3.metric("Некроз (отмирание)", f"{m.necrosis_ratio*100:.1f}%")

                st.markdown("---")
                st.markdown("#### Протокол лечения:")
                st.info(f"💊 **Терапия:** {result.treatment}")
                st.warning(f"🛡️ **Профилактика:** {result.prevention}")

                st.markdown("#### Топ альтернативных диагнозов:")
                for i, cand in enumerate(result.top_candidates[:3], 1):
                    st.write(f"{i}. {cand['crop_ru']} - {cand['disease_ru']}: {cand['confidence']*100:.1f}%")

    # 3. Knowledge Base
    elif mode == "📚 Справочник болезней (38)":
        st.subheader("Агрономическая база знаний (38 классов PlantVillage)")
        st.markdown("Полный перечень поддерживаемых культур и фитопатологий с протоколами лечения:")

        search_query = st.text_input("Поиск по культуре или болезни", "").lower()

        for class_name, data in AGRONOMIC_KNOWLEDGE_BASE.items():
            if search_query:
                combined_text = f"{data['crop']} {data['disease_ru']} {class_name} {data['pathogen']}".lower()
                if search_query not in combined_text:
                    continue

            with st.expander(f"{'✓' if data['is_healthy'] else '⚠️'} {data['crop']}: {data['disease_ru']} ({class_name})"):
                st.write(f"**Культура:** {data['crop']} ({data['crop_en']})")
                st.write(f"**Патоген:** {data['pathogen']}")
                st.write(f"**Тяжесть:** {data['severity']}")
                st.write(f"**Лечение:** {data['treatment']}")
                st.write(f"**Профилактика:** {data['prevention']}")


if __name__ == "__main__":
    main()
