import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import base64
import os
import time
import pandas as pd
from datetime import datetime
from model_loader import load_unet_model, load_classifier_model

# ... (جميع التعريفات السابقة مثل LANGUAGES, t(), CSS, session state, sidebar, etc.) ...

# ========== دوال معالجة الصور (قائمة) ==========
def clean_mask(mask, min_area=500):
    """تنظيف الماسك بإزالة الأجسام الصغيرة وتطبيق عمليات مورفولوجية."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(cleaned, [c], -1, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned

def extract_best_conjunctiva(image, raw_mask):
    """استخراج الملتحمة باستخدام الماسك النظيف (بدون تحسين ألوان)."""
    mask_clean = clean_mask(raw_mask)
    conj = cv2.bitwise_and(image, image, mask=mask_clean)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        bbox = (x, y, w, h)
    else:
        bbox = (0, 0, 0, 0)
    return conj, mask_clean, bbox

def predict_anemia(model, image, device):
    """تصنيف الصورة باستخدام النموذج (EfficientNet)."""
    img_resized = cv2.resize(image, (224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.sigmoid(out).item()
    if prob > 0.5:
        result = "Anemic"
        confidence = prob * 100
    else:
        result = "Non Anemic"
        confidence = (1 - prob) * 100
    return result, confidence, prob

def calculate_hb(conjunctiva_image, mask):
    """
    حساب قيمة Hb التقريبية باستخدام المعادلات من الورقة.
    المتوسطات تحسب فقط على البكسلات التي يغطيها الماسك (المنطقة المقطعة).
    """
    # استخراج البكسلات المقنعة فقط
    masked_pixels = conjunctiva_image[mask > 0]
    if masked_pixels.size == 0:
        return None, None, None  # لا توجد بيانات كافية
    
    # حساب متوسطات القنوات
    r_mean = np.mean(masked_pixels[:, 0]) / 255.0  # تطبيع إلى [0,1]
    g_mean = np.mean(masked_pixels[:, 1]) / 255.0
    b_mean = np.mean(masked_pixels[:, 2]) / 255.0
    
    # المعادلة (1): Hb_raw = sigmoid(-1.922 + 0.206*r - 0.241*g + 0.012*b)
    z = -1.922 + 0.206 * r_mean - 0.241 * g_mean + 0.012 * b_mean
    hb_raw = 1 / (1 + np.exp(-z))  # في المجال [0,1]
    
    # المعادلة (2): تحويل خطي من [0,1] إلى [7,15]
    a, b = 0.0, 1.0
    c, d = 7.0, 15.0
    hb_gdl = c + (d - c) * (hb_raw - a) / (b - a)  # في المجال [7,15] g/dL
    
    # تصنيف حسب عتبة WHO (11 g/dL)
    if hb_gdl < 11:
        hb_diagnosis = "Anemic (Hb < 11)"
    else:
        hb_diagnosis = "Non Anemic (Hb ≥ 11)"
    
    return hb_gdl, hb_raw, hb_diagnosis

@st.cache_resource
def load_models():
    unet, dev_unet = load_unet_model()
    clf, dev_clf = load_classifier_model()
    return unet, dev_unet, clf, dev_clf

# ... (باقي الكود: upload, preview, button) ...

# داخل زر التحليل بعد الحصول على conj_enhanced و final_mask:
if st.button(t("analyze_btn"), use_container_width=True):
    with st.spinner(t("loading_models")):
        unet_model, unet_device, clf_model, clf_device = load_models()

    progress_bar = st.progress(0)
    for i in range(10):
        time.sleep(0.05)
        progress_bar.progress((i+1)*10)

    with st.spinner(t("analyzing")):
        img = np.array(Image.open(uploaded).convert('RGB'))
        img = cv2.flip(img, 1)

        # تجزئة U‑Net
        transform_unet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform_unet(img).unsqueeze(0).to(unet_device)
        with torch.no_grad():
            raw_mask = torch.sigmoid(unet_model(tensor)).squeeze().cpu().numpy()
            raw_mask = (raw_mask > 0.5).astype(np.uint8) * 255
            raw_mask = cv2.resize(raw_mask, (img.shape[1], img.shape[0]))

        conj_enhanced, final_mask, bbox = extract_best_conjunctiva(img, raw_mask)

        # تصنيف النموذج الأساسي
        result, confidence, raw_pred = predict_anemia(clf_model, conj_enhanced, clf_device)
        anemia_pct = raw_pred * 100
        non_pct = (1 - raw_pred) * 100

        # ---- حساب Hb (الإضافة الجديدة) ----
        hb_gdl, hb_raw, hb_diagnosis = calculate_hb(conj_enhanced, final_mask)
        # hb_gdl قد تكون None إذا كانت المنطقة فارغة، نتعامل معها

        progress_bar.empty()
        st.success(t("analysis_done"))

        # ===== عرض النتائج =====
        st.markdown(f'<div class="section-title">{t("results_title")}</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**{t('result_original')}**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown(f"**{t('result_mask')}**")
            st.image(final_mask, use_container_width=True, clamp=True)
        with col3:
            st.markdown(f"**{t('result_conjunctiva')}**")
            st.image(conj_enhanced, use_container_width=True)

        # متريكات التنظيف
        before = np.sum(raw_mask > 0) / 255
        after = np.sum(final_mask > 0) / 255
        reduction = ((before - after) / before * 100) if before > 0 else 0

        m1, m2 = st.columns(2)
        with m1:
            st.metric(t("metric_surface"), f"{after:.0f} px²")
        with m2:
            st.metric(t("metric_cleaning"), f"{reduction:.1f}%")

        # التشخيص الأساسي
        st.markdown(f'<div class="section-title">{t("diagnostic_title")}</div>', unsafe_allow_html=True)
        col_res, col_conf = st.columns(2)
        with col_res:
            if result == "Anemic":
                st.markdown(f"""
                <div class="result-card positive">
                    <div class="badge-ic">🩸</div>
                    <div>
                        <h2>{t('diagnostic_anemic')}</h2>
                        <div class="confidence">{t('diagnostic_confidence')} : <strong>{confidence:.1f}%</strong></div>
                        <div class="sub">{t('diagnostic_anemic_desc')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card negative">
                    <div class="badge-ic">✅</div>
                    <div>
                        <h2>{t('diagnostic_non_anemic')}</h2>
                        <div class="confidence">{t('diagnostic_confidence')} : <strong>{confidence:.1f}%</strong></div>
                        <div class="sub">{t('diagnostic_non_anemic_desc')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with col_conf:
            st.metric(t("diagnostic_confidence"), f"{confidence:.1f}%")
            st.progress(int(confidence))

        # ---- عرض قيمة Hb (الإضافة الجديدة) ----
        st.markdown("---")
        st.markdown("#### 🧪 تقدير الهيموغلوبين (Hb) التقريبي")
        if hb_gdl is not None:
            col_hb1, col_hb2, col_hb3 = st.columns(3)
            with col_hb1:
                st.metric("قيمة Hb المحسوبة", f"{hb_gdl:.2f} g/dL")
            with col_hb2:
                st.metric("القيمة الخام (0-1)", f"{hb_raw:.4f}")
            with col_hb3:
                st.metric("التشخيص حسب Hb", hb_diagnosis)
            
            # إضافة مؤشر بصري مقارنة مع العتبة 11
            import plotly.graph_objects as go
            fig_hb = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = hb_gdl,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Hb (g/dL)"},
                delta = {'reference': 11, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [7, 15], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [7, 11], 'color': 'lightcoral'},
                        {'range': [11, 15], 'color': 'lightgreen'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 11
                    }
                }
            ))
            fig_hb.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_hb, use_container_width=True)
        else:
            st.warning("لم يتمكن النظام من حساب Hb بسبب عدم كفاية المنطقة المقطعة.")

        # ... (باقي الكود: رسم بياني للاحتمالات، السجل، التفاصيل التقنية، إلخ) ...
