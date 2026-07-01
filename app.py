import streamlit as st
import torch
import cv2
from PIL import Image
import torch.nn as nn
import numpy as np
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import os
from model_loader import load_unet_model, load_classifier_model

# ========== إعداد الصفحة ==========
st.set_page_config(
    page_title="AnemiCheck - Détection d'Anémie",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== CSS (النمط الأصلي البسيط) ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseBorder {
        0%, 100% { border-color: #e11d48; box-shadow: 0 0 0 0 rgba(225,29,72,0.20); }
        50% { border-color: #fb7185; box-shadow: 0 0 0 12px rgba(225,29,72,0); }
    }
    @keyframes bounceSlow {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.85); }
        70% { opacity: 1; transform: scale(1.03); }
        100% { opacity: 1; transform: scale(1); }
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(160deg, #fff8f7 0%, #fff1f1 50%, #ffeaea 100%);
    }

    #MainMenu, footer {visibility: hidden;}
    .fade-in-up { animation: fadeInUp 0.7s ease both; }

    /* ===== منطقة الرفع ===== */
    .upload-zone {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(10px);
        border: 2.5px dashed #e11d48;
        border-radius: 28px;
        padding: 38px 20px;
        text-align: center;
        margin: 15px 0 5px 0;
        animation: pulseBorder 2.5s ease-in-out infinite, fadeInUp 0.7s ease both;
    }
    .upload-zone .upload-icon {
        font-size: 54px;
        animation: bounceSlow 2.2s ease-in-out infinite;
        display: inline-block;
    }
    .upload-zone h4 {
        color: #9f1239;
        margin: 10px 0 5px 0;
        font-weight: 700;
    }
    .upload-zone p {
        color: #6b7280;
        font-size: 14px;
        margin: 0;
    }

    .logo-container {
        text-align: center;
        padding: 15px;
        margin-bottom: 5px;
    }
    .logo-image {
        width: 100%;
        max-width: 260px;
        height: auto;
        object-fit: contain;
        margin: 0 auto;
        display: block;
        filter: drop-shadow(0 12px 20px rgba(225,29,72,0.18));
    }
    .app-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        background: linear-gradient(135deg, #9f1239, #e11d48, #fb7185);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 5px 0;
        letter-spacing: 1px;
    }
    .app-subtitle-ar {
        text-align: center;
        font-size: 24px;
        color: #9f1239;
        margin: 5px 0;
        font-weight: 700;
        font-family: 'Tajawal', sans-serif;
    }
    .app-subtitle {
        text-align: center;
        font-size: 17px;
        color: #e11d48;
        margin: 5px 0;
        font-weight: 500;
    }
    .app-tagline {
        text-align: center;
        font-size: 14px;
        color: #6b7280;
        margin-top: 5px;
        padding: 8px 22px;
        background: rgba(255,255,255,0.6);
        border-radius: 30px;
        border: 1px solid rgba(225,29,72,0.25);
        display: inline-block;
    }
    .tagline-container { text-align: center; margin-bottom: 15px; }

    /* ===== QR Code (بدون رابط نصي) ===== */
    .qr-card {
        background: rgba(255,255,255,0.85);
        border-radius: 24px;
        padding: 20px 26px;
        display: flex;
        align-items: center;
        gap: 18px;
        box-shadow: 0 14px 28px -10px rgba(225,29,72,0.16);
        border: 1px solid rgba(225,29,72,0.15);
        max-width: 400px;
        margin: 18px auto;
        transition: all 0.3s ease;
    }
    .qr-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 32px -10px rgba(225,29,72,0.22);
    }
    .qr-card img {
        border-radius: 14px;
        border: 1px solid rgba(225,29,72,0.15);
        background: white;
        padding: 6px;
        flex-shrink: 0;
    }
    .qr-text h5 {
        margin: 0 0 6px 0;
        color: #9f1239;
        font-size: 16px;
        font-weight: 700;
    }
    .qr-text p {
        margin: 0;
        font-size: 13px;
        color: #6b7280;
        line-height: 1.5;
    }
    /* تم إزالة أي styling للرابط النصي */

    /* ===== النتائج ===== */
    .result-anemia {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe1e4 100%);
        border-radius: 22px;
        padding: 26px;
        margin: 20px 0;
        border: 2px solid #e11d48;
        box-shadow: 0 14px 28px rgba(225,29,72,0.14);
        animation: popIn 0.5s ease both;
    }
    .result-non-anemia {
        background: linear-gradient(135deg, #f0fdf6 0%, #dcfce7 100%);
        border-radius: 22px;
        padding: 26px;
        margin: 20px 0;
        border: 2px solid #16a34a;
        box-shadow: 0 14px 28px rgba(22,163,74,0.14);
        animation: popIn 0.5s ease both;
    }

    .stButton > button {
        background: linear-gradient(90deg, #e11d48, #f43f5e, #fb7185);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 14px 35px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 14px rgba(225,29,72,0.3);
    }
    .stButton > button:hover {
        transform: scale(1.02) translateY(-1px);
        background: linear-gradient(90deg, #9f1239, #e11d48, #f43f5e);
        box-shadow: 0 8px 18px rgba(225,29,72,0.35);
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        color: #9f1239;
        margin: 20px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #e11d48;
        display: inline-block;
    }
    .section-container { text-align: center; margin-bottom: 20px; }

    /* ===== تنويه طبي بسيط ===== */
    .disclaimer {
        background: rgba(255,255,255,0.7);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        font-size: 12.5px;
        color: #495057;
        margin-top: 30px;
        border: 1px solid rgba(225,29,72,0.2);
        border-left: 4px solid #e11d48;
    }

    .stRadio > div { gap: 30px; justify-content: center; }
    .stRadio label { font-size: 16px; font-weight: 500; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #e11d48; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.7);
        border-radius: 16px;
        padding: 12px 10px;
        border: 1px solid rgba(225,29,72,0.12);
    }
</style>
""", unsafe_allow_html=True)

# ========== عرض الشعار ==========
def get_logo_base64():
    possible_names = ["logo.png", "logo.jpg", "logo.jpeg", "LOGO.png", "Logo.png", "logo.PNG"]
    for logo_path in possible_names:
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode(), logo_path
    return None, None

logo_base64, _ = get_logo_base64()
if logo_base64:
    st.markdown(f"""
    <div class="logo-container fade-in-up">
        <img src="data:image/png;base64,{logo_base64}" class="logo-image">
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="logo-container fade-in-up">
        <div style="font-size: 150px; text-align: center;">🩸👁️</div>
    </div>
    """, unsafe_allow_html=True)

# عنوان التطبيق
st.markdown("""
<div class="app-title fade-in-up">AnemiCheck</div>
<div class="app-subtitle-ar fade-in-up">رؤية ميد</div>
<div class="app-subtitle fade-in-up">Détection d'Anémie par Intelligence Artificielle</div>
<div class="tagline-container fade-in-up">
    <div class="app-tagline">فحص فقر الدم غير الجراحي • Non-Invasive Anemia Screening</div>
</div>
""", unsafe_allow_html=True)

# ========== QR Code (يظهر بدون رابط نصي) ==========
APP_URL = "https://hwaxrexkahkxaazwwjjr3d.streamlit.app/"
qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&margin=8&color=9f1239&data={APP_URL}"

st.markdown(f"""
<div class="qr-card fade-in-up">
    <img src="{qr_api_url}" width="110" height="110" alt="QR Code">
    <div class="qr-text">
        <h5>📱 شارك التطبيق</h5>
        <p>امسح الكود بالكاميرا للوصول السريع</p>
        <!-- تم حذف الرابط النصي <a> نهائياً -->
    </div>
</div>
""", unsafe_allow_html=True)

# ========== منطقة رفع الصورة ==========
st.markdown("---")
st.markdown('<div class="section-container fade-in-up"><span class="section-title">📸 Acquisition de l\'image</span></div>', unsafe_allow_html=True)

st.markdown("""
<div class="upload-zone">
    <div class="upload-icon">👁️📤</div>
    <h4>حط الصورة تاعك هنا</h4>
    <p>صورة واضحة للعين، أو تصور بالكاميرا مباشرة</p>
</div>
""", unsafe_allow_html=True)

option = st.radio(
    "Choisissez la méthode / اختر الطريقة:",
    ["📁 Télécharger une image", "📷 Prendre une photo"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded = None
if option == "📁 Télécharger une image":
    uploaded = st.file_uploader("Téléchargez une image de l'œil", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
else:
    uploaded = st.camera_input("📷 Prenez une photo de l'œil", disabled=False, label_visibility="collapsed")

# ========== دوال المعالجة (مع CLAHE) ==========
def clean_mask(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(clean, [contour], -1, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    return clean

def enhance_conjunctiva(image):
    """CLAHE pour améliorer le contraste de la conjonctive"""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        return enhanced
    return image

def extract_best_conjunctiva(img, mask):
    mask = clean_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 15
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        if w > 0 and h > 0:
            cropped = img[y:y+h, x:x+w]
            cropped_enhanced = enhance_conjunctiva(cropped)
            return cropped_enhanced, mask, (x, y, w, h)
    conjunctiva = cv2.bitwise_and(img, img, mask=mask)
    conjunctiva_enhanced = enhance_conjunctiva(conjunctiva)
    return conjunctiva_enhanced, mask, None

# ========== دالة التصنيف (بدون أي تلاعب، النتيجة الأصلية) ==========
def predict_anemia(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.sigmoid(output).item()   # القيمة الخام بين 0 و 1

    # نستخدم prediction مباشرة دون قلب أو تلاعب
    if prediction >= 0.5:
        result = "Anemic"
        confidence = prediction * 100
    else:
        result = "Non Anemic"
        confidence = (1 - prediction) * 100

    return result, confidence, prediction

# ========== معالجة الصورة ==========
if uploaded is not None:
    with st.spinner("🔃 Chargement des modèles..."):
        try:
            unet_model, unet_device = load_unet_model()
        except Exception as e:
            st.error(f"❌ Erreur U-Net: {e}")
            st.stop()
        try:
            classifier_model, classifier_device = load_classifier_model()
        except Exception as e:
            st.error(f"❌ Erreur EfficientNet-B3: {e}")
            st.stop()

    with st.spinner("🔍 Analyse en cours..."):
        img = np.array(Image.open(uploaded).convert('RGB'))
        img = cv2.flip(img, 1)  # تصحيح انعكاس الكاميرا (إذا لزم الأمر)

        # Segmentation
        transform_unet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        img_tensor = transform_unet(img).unsqueeze(0).to(unet_device)
        with torch.no_grad():
            raw_mask = torch.sigmoid(unet_model(img_tensor)).squeeze().cpu().numpy()
            raw_mask = (raw_mask > 0.5).astype(np.uint8) * 255
            raw_mask = cv2.resize(raw_mask, (img.shape[1], img.shape[0]))

        cleaned_mask = clean_mask(raw_mask)
        conjunctiva, final_mask, _ = extract_best_conjunctiva(img, cleaned_mask)

        # Classification (نتيجة النموذج الحقيقية)
        result, confidence, raw_pred = predict_anemia(classifier_model, conjunctiva, classifier_device)

        # حساب النسب للأعمدة
        anemia_percent = raw_pred * 100
        non_anemia_percent = (1 - raw_pred) * 100

        st.success("✅ Analyse terminée")

        st.markdown("## 📊 Résultats")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🖼️ Originale**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("**🎭 Segmentation**")
            st.image(final_mask, use_container_width=True, clamp=True)
        with col3:
            st.markdown("**👁️ Conjonctive (CLAHE)**")
            st.image(conjunctiva, use_container_width=True)

        before_area = np.sum(raw_mask > 0) / 255
        after_area = np.sum(final_mask > 0) / 255
        reduction = ((before_area - after_area) / before_area * 100) if before_area > 0 else 0

        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Surface segmentée", f"{after_area:.0f} px²")
        with col_stat2:
            st.metric("Nettoyage", f"{reduction:.1f}% d'artefacts")

        st.markdown("---")
        st.markdown("## 🩺 Diagnostic")
        col_result, col_conf = st.columns(2)
        with col_result:
            if result == "Anemic":
                st.markdown("""
                <div class="result-anemia">
                    <h2 style="color: #dc2626; font-size: 32px;">🩸 Anémie</h2>
                    <p style="font-size: 18px;"><b>Anémie détectée</b></p>
                    <p style="font-size: 14px; color: #666;">يوجد فقر دم</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-non-anemia">
                    <h2 style="color: #16a34a; font-size: 32px;">✅ Non Anémie</h2>
                    <p style="font-size: 18px;"><b>Pas d'anémie détectée</b></p>
                    <p style="font-size: 14px; color: #666;">لا يوجد فقر دم</p>
                </div>
                """, unsafe_allow_html=True)
        with col_conf:
            st.metric("Confiance", f"{confidence:.1f}%")
            st.progress(int(confidence))

        st.markdown("### 📈 Niveau d'anémie")
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ['Non anemic', 'Anemic']
        values = [non_anemia_percent, anemia_percent]  # النسب الحقيقية
        colors = ['#10b981', '#dc2626']
        bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='white', linewidth=2)
        ax.set_ylim([0, 100])
        ax.set_ylabel('Pourcentage (%)', fontsize=12)
        ax.set_title('Probabilité d\'anémie (résultat brut du modèle)', fontsize=14, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', fontweight='bold', fontsize=14,
                    color='#8b0000' if val == anemia_percent else '#166534')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

        with st.expander("📈 Détails techniques"):
            st.write(f"**Diagnostic:** {result}")
            st.write(f"**Confiance:** {confidence:.2f}%")
            st.write(f"**Valeur sigmoïde (brute):** {raw_pred:.4f}")
            st.write(f"**Probabilité Anémie:** {anemia_percent:.1f}%")
            st.write(f"**Probabilité Non Anémie:** {non_anemia_percent:.1f}%")
            st.write(f"**Appareil:** {'GPU' if classifier_device.type == 'cuda' else 'CPU'}")
            st.write("**Amélioration:** CLAHE sur la conjonctive")
            st.write("**Décision:** 100% basée sur le modèle (aucune correction manuelle)")

        # تنويه طبي بسيط
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <b>Avertissement médical / تنويه طبي</b><br>
            Ce résultat est fourni à titre indicatif et ne remplace pas un avis médical professionnel.<br>
            Veuillez consulter un médecin pour un diagnostic fiable.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 40px; background: white; border-radius: 24px; margin: 20px 0;">
        <div style="font-size: 48px; margin-bottom: 20px;">📸</div>
        <h3>Bienvenue sur AnemiCheck</h3>
        <p>Sélectionnez une méthode d'acquisition ci-dessus pour commencer</p>
    </div>
    """, unsafe_allow_html=True)

# ========== Guide d'utilisation ==========
with st.expander("ℹ️ Guide d'utilisation"):
    st.markdown("""
    **Comment fonctionne l'application ?**
    
    1. **U-Net** : Segmentation avancée pour extraire la conjonctive.
    2. **EfficientNet-B3** : Classification pour détecter l'anémie.
    3. **CLAHE** : Améliore le contraste de la conjonctive pour une meilleure précision.
    
    **Conseils :**
    - 📷 Utilisez une image claire et bien éclairée.
    - 👁️ Assurez-vous que l'œil est bien visible.
    - 🩸 Les résultats sont à titre indicatif.
    """)
