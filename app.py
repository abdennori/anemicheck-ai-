import streamlit as st
import torch
import cv2
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
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

# ========== CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Tajawal:wght@400;500;700;800&display=swap');

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
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
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', 'Tajawal', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(251,113,133,0.16) 0%, transparent 45%),
            radial-gradient(circle at 85% 25%, rgba(244,63,94,0.12) 0%, transparent 45%),
            linear-gradient(160deg, #fff8f7 0%, #fff1f1 50%, #ffeaea 100%);
        background-size: 200% 200%;
        animation: gradientShift 18s ease infinite;
    }

    #MainMenu, footer {visibility: hidden;}
    .fade-in-up { animation: fadeInUp 0.7s ease both; }

    /* ===== الهيرو الخاص بالطبيب ===== */
    .doctor-hero {
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 32px;
        padding: 35px 30px 30px;
        margin: 15px 0 20px 0;
        text-align: center;
        border: 1px solid rgba(225,29,72,0.12);
        box-shadow: 0 20px 40px -12px rgba(225,29,72,0.15);
        transition: all 0.3s ease;
    }
    .doctor-hero:hover {
        box-shadow: 0 28px 48px -12px rgba(225,29,72,0.20);
    }
    .doctor-icon {
        font-size: 64px;
        margin-bottom: 6px;
        display: block;
        animation: bounceSlow 2.8s ease-in-out infinite;
    }
    .doctor-title {
        font-size: 38px;
        font-weight: 800;
        color: #1f2937;
        margin: 0 0 4px 0;
        font-family: 'Tajawal', sans-serif;
        letter-spacing: 1px;
    }
    .doctor-title span {
        background: linear-gradient(135deg, #9f1239, #e11d48);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .doctor-subtitle {
        font-size: 20px;
        font-weight: 700;
        color: #9f1239;
        margin: 0 0 2px 0;
        font-family: 'Tajawal', sans-serif;
    }
    .doctor-subtitle-light {
        font-size: 16px;
        color: #6b7280;
        margin: 0 0 18px 0;
        font-weight: 500;
    }
    .doctor-features {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
        margin: 20px 0;
    }
    .feature-item {
        background: rgba(255,255,255,0.8);
        border: 1px solid rgba(225,29,72,0.15);
        border-radius: 20px;
        padding: 14px 22px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 15px;
        font-weight: 600;
        color: #1f2937;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .feature-item:hover {
        transform: translateY(-3px);
        border-color: #e11d48;
        box-shadow: 0 8px 20px rgba(225,29,72,0.10);
    }
    .feature-item .badge {
        background: #f59e0b;
        color: white;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 12px;
        border-radius: 30px;
        margin-left: 6px;
        letter-spacing: 0.5px;
    }
    .doctor-cta {
        background: linear-gradient(90deg, #e11d48, #f43f5e, #fb7185);
        color: white;
        border: none;
        border-radius: 60px;
        padding: 16px 48px;
        font-size: 20px;
        font-weight: 700;
        font-family: 'Tajawal', sans-serif;
        cursor: not-allowed;
        opacity: 0.85;
        box-shadow: 0 8px 24px rgba(225,29,72,0.25);
        transition: all 0.3s ease;
        margin: 10px 0 6px 0;
        display: inline-block;
        letter-spacing: 1px;
    }
    .doctor-cta:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 28px rgba(225,29,72,0.35);
    }
    .doctor-brand {
        font-size: 13px;
        color: #9ca3af;
        margin-top: 12px;
        font-weight: 500;
        letter-spacing: 2px;
        border-top: 1px solid #f3f4f6;
        padding-top: 12px;
    }
    .doctor-brand strong {
        color: #e11d48;
    }

    /* ===== باقي العناصر ===== */
    .upload-zone {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
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
        background: transparent;
        border-radius: 30px;
        margin-bottom: 5px;
    }
    .logo-image {
        width: 100%;
        max-width: 260px;
        height: auto;
        object-fit: contain;
        margin: 0 auto;
        display: block;
        transition: transform 0.4s ease;
        filter: drop-shadow(0 12px 20px rgba(225,29,72,0.18));
    }
    .logo-image:hover {
        transform: scale(1.04) rotate(-1deg);
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
        width: auto;
    }
    .tagline-container { text-align: center; margin-bottom: 15px; }

    .qr-card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(8px);
        border-radius: 24px;
        padding: 20px 26px;
        display: flex;
        align-items: center;
        gap: 18px;
        box-shadow: 0 14px 28px -10px rgba(225,29,72,0.16);
        border: 1px solid rgba(225,29,72,0.15);
        max-width: 520px;
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
        margin: 0 0 6px 0;
        font-size: 12.5px;
        color: #6b7280;
        line-height: 1.5;
    }
    .qr-text a {
        color: #e11d48;
        font-weight: 600;
        text-decoration: none;
        word-break: break-all;
        font-size: 13px;
    }
    .qr-text .copy-btn {
        margin-top: 8px;
        background: #e11d48;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 6px 18px;
        cursor: pointer;
        font-size: 13px;
        transition: background 0.2s;
    }
    .qr-text .copy-btn:hover {
        background: #9f1239;
    }

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

    /* ===== تحسين التنويه الطبي (نتائج أولية) ===== */
    .disclaimer {
        background: #fffbeb;
        border-radius: 16px;
        padding: 18px 20px;
        text-align: center;
        font-size: 13px;
        color: #4b5563;
        margin-top: 30px;
        border: 1px solid #fcd34d;
        border-left: 6px solid #f59e0b;
        box-shadow: 0 4px 12px rgba(245,158,11,0.12);
        transition: all 0.3s ease;
    }
    .disclaimer:hover {
        box-shadow: 0 8px 20px rgba(245,158,11,0.18);
        transform: translateY(-2px);
    }
    .disclaimer strong {
        color: #b45309;
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

# ========== JavaScript لطلب الكاميرا ==========
st.markdown("""
<script>
    async function requestCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            console.log("Camera access granted");
            stream.getTracks().forEach(track => track.stop());
        } catch (err) {
            console.log("Camera access denied: " + err);
        }
    }
    requestCamera();
</script>
""", unsafe_allow_html=True)

# ========== عرض الشعار ==========
def get_logo_base64():
    possible_names = ["logo.png", "logo.jpg", "logo.jpeg", "LOGO.png", "Logo.png", "logo.PNG"]
    for logo_path in possible_names:
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode(), logo_path
    return None, None

logo_base64, logo_name = get_logo_base64()

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

# ============================================================
# ====== HERO SECTION: تطبيق مع طبيب ======
# ============================================================
st.markdown("""
<div class="doctor-hero fade-in-up">
    <span class="doctor-icon">🩺</span>
    <h1 class="doctor-title">تطبيق مع <span>طبيب</span></h1>
    <p class="doctor-subtitle">طبيبك معك بأقبل من دقيقتين</p>
    <p class="doctor-subtitle-light">بدون مواعيد أو انتظار • Sans rendez-vous ni attente</p>
    
    <div class="doctor-features">
        <div class="feature-item">💬 محادثة نصية <span class="badge">قريباً</span></div>
        <div class="feature-item">📞 مكالمة صوتية <span class="badge">قريباً</span></div>
        <div class="feature-item">🖼️ إرسال صورة <span class="badge">قريباً</span></div>
    </div>

    <button class="doctor-cta" disabled>🔜 تحدث مع طبيب الآن</button>
    <div class="doctor-brand">الطبيب <strong>altibbi</strong></div>
</div>
""", unsafe_allow_html=True)

# ========== مشاركة التطبيق عبر QR Code (محسّن) ==========
APP_URL = "https://hwaxrexkahkxaazwwjjr3d.streamlit.app/"
qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&margin=10&color=9f1239&data={APP_URL}"

st.markdown(f"""
<div class="qr-card fade-in-up">
    <img src="{qr_api_url}" width="120" height="120" alt="QR Code pour AnemiCheck" loading="lazy" 
         onerror="this.style.display='none'; document.getElementById('qr-fallback').style.display='block';">
    <div id="qr-fallback" style="display:none; font-size:14px; color:#e11d48;">⚠️ QR Code non disponible</div>
    <div class="qr-text">
        <h5>📱 شارك التطبيق / Partager l'app</h5>
        <p>امسح الكود بالكاميرا باش تفتح التطبيق فأي جهاز آخر مباشرة</p>
        <a href="{APP_URL}" target="_blank">{APP_URL}</a>
        <br>
        <button class="copy-btn" onclick="navigator.clipboard.writeText('{APP_URL}')">📋 نسخ الرابط</button>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== منطقة رفع/تصوير الصورة ==========
st.markdown("---")
st.markdown('<div class="section-container fade-in-up"><span class="section-title">📸 Méthode d\'acquisition / طريقة الحصول على الصورة</span></div>', unsafe_allow_html=True)

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

# ========== دوال المعالجة ==========
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

# ========== دالة التصنيف (بدون تصحيح) ==========
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
        prediction = torch.sigmoid(output).item()
    if prediction >= 0.5:
        result = "Anemic"
        confidence = prediction * 100
    else:
        result = "Non Anemic"
        confidence = (1 - prediction) * 100
    return result, confidence, prediction

# ========== معالجة الصورة ==========
if uploaded is not None:
    with st.spinner("🔃 Chargement des modèles intelligents..."):
        try:
            unet_model, unet_device = load_unet_model()
        except Exception as e:
            st.error(f"❌ Erreur de chargement U-Net: {e}")
            st.stop()
        try:
            classifier_model, classifier_device = load_classifier_model()
        except Exception as e:
            st.error(f"❌ Erreur de chargement EfficientNet-B3: {e}")
            st.stop()

    with st.spinner("🔍 Analyse de l'image en cours..."):
        img = np.array(Image.open(uploaded).convert('RGB'))
        img = cv2.flip(img, 1)

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
        conjunctiva, final_mask, bbox = extract_best_conjunctiva(img, cleaned_mask)

        result, confidence, raw_pred = predict_anemia(classifier_model, conjunctiva, classifier_device)

        anemia_percent = raw_pred * 100
        non_anemia_percent = (1 - raw_pred) * 100

        st.success("✅ Analyse terminée avec succès")

        st.markdown("## 📊 Résultats de l'analyse")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🖼️ Image originale**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("**🎭 Segmentation (U-Net)**")
            st.image(final_mask, use_container_width=True, clamp=True)
        with col3:
            st.markdown("**👁️ Conjonctive extraite (améliorée CLAHE)**")
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
        st.markdown("## 🩺 Diagnostic (EfficientNet-B3)")
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
            st.metric("Niveau de confiance", f"{confidence:.1f}%")
            st.progress(int(confidence))

        st.markdown("### 📈 Niveau d'anémie")
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ['Non anemic', 'Anemic']
        values = [non_anemia_percent, anemia_percent]
        colors = ['#10b981', '#dc2626']
        bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='white', linewidth=2)
        ax.set_ylim([0, 100])
        ax.set_ylabel('Pourcentage (%)', fontsize=12)
        ax.set_title('Probabilité d\'anémie (résultat brut)', fontsize=14, fontweight='bold')
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
            st.write(f"**Niveau de confiance:** {confidence:.2f}%")
            st.write(f"**Valeur brute (sigmoid):** {raw_pred:.4f}")
            st.write(f"**Règle utilisée:** Pas de correction, résultat direct du modèle")
            st.write(f"**Anémie (probabilité):** {anemia_percent:.1f}%")
            st.write(f"**Non Anémie (probabilité):** {non_anemia_percent:.1f}%")
            st.write(f"**Appareil utilisé:** {'GPU' if classifier_device.type == 'cuda' else 'CPU'}")
            st.write(f"**Modèle de segmentation:** U-Net (ResNet34)")
            st.write(f"**Modèle de classification:** EfficientNet-B3")
            st.write("**Amélioration appliquée:** CLAHE sur la conjonctive (améliore le contraste)")

        # ========== التنويه المحسّن (نتائج أولية) ==========
        st.markdown("""
        <div class="disclaimer">
            <div style="display: flex; align-items: center; gap: 10px; justify-content: center; margin-bottom: 8px;">
                <span style="font-size: 28px;">⚠️</span>
                <span style="font-size: 18px; font-weight: 700; color: #b45309;">Résultats préliminaires / نتائج أولية</span>
            </div>
            <p style="margin: 0; font-size: 14px; color: #4b5563; line-height: 1.6;">
                Ces résultats sont générés par un modèle d'intelligence artificielle et sont fournis à titre indicatif.<br>
                Ils peuvent comporter des erreurs et ne remplacent en aucun cas un diagnostic médical professionnel.<br>
                <strong>Veuillez consulter un médecin pour une évaluation fiable.</strong>
            </p>
            <p style="margin: 8px 0 0 0; font-size: 13px; color: #6b7280; line-height: 1.5;">
                هذه النتائج مبنية على الذكاء الاصطناعي وهي لأغراض إرشادية فقط، وقد تحتوي على أخطاء.<br>
                لا تغني عن استشارة الطبيب المختص للحصول على تشخيص دقيق.
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="fade-in-up" style="text-align: center; padding: 40px; background: white; border-radius: 24px; margin: 20px 0;">
        <div style="font-size: 48px; margin-bottom: 20px;">📸</div>
        <h3>Bienvenue sur AnemicCheck</h3>
        <p>Sélectionnez une méthode d'acquisition ci-dessus pour commencer l'analyse</p>
    </div>
    """, unsafe_allow_html=True)

# ========== Guide d'utilisation ==========
with st.expander("ℹ️ Guide d'utilisation / كيف تستخدم التطبيق"):
    st.markdown("""
    **Comment fonctionne l'application ? / كيف يعمل التطبيق ؟**
    
    1. **U-Net** : Segmentation avancée pour extraire la conjonctive (الملتحمة)
    2. **EfficientNet-B3** : Classification pour détecter l'anémie (تصنيف للكشف عن فقر الدم)
    3. **Amélioration CLAHE** : Améliore le contraste de la conjonctive pour une meilleure précision (تحسين التباين لزيادة الدقة)
    
    **Conseils pour une analyse optimale / نصائح للحصول على تحليل دقيق :**
    - 📷 Utilisez une image claire et bien éclairée / استخدم صورة واضحة ومضاءة جيداً
    - 👁️ Assurez-vous que l'œil est bien visible / تأكد من أن العين ظاهرة بوضوح
    - 🩸 Les résultats sont à titre indicatif / النتائج هي لأغراض إرشادية فقط
    """)
