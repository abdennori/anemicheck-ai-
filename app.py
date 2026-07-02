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
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes pulseBorder { 0%, 100% { border-color: #dc2626; box-shadow: 0 0 0 0 rgba(220,38,38,0.25); } 50% { border-color: #f87171; box-shadow: 0 0 0 10px rgba(220,38,38,0); } }
    @keyframes bounceSlow { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
    @keyframes popIn { 0% { opacity: 0; transform: scale(0.85); } 70% { opacity: 1; transform: scale(1.03); } 100% { opacity: 1; transform: scale(1); } }
    .stApp { background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%); }
    .fade-in-up { animation: fadeInUp 0.7s ease both; }
    .upload-zone { background: white; border: 2.5px dashed #dc2626; border-radius: 24px; padding: 35px 20px; text-align: center; margin: 15px 0 5px 0; animation: pulseBorder 2.5s ease-in-out infinite, fadeInUp 0.7s ease both; }
    .upload-zone .upload-icon { font-size: 54px; animation: bounceSlow 2.2s ease-in-out infinite; display: inline-block; }
    .upload-zone h4 { color: #b91c1c; margin: 10px 0 5px 0; }
    .upload-zone p { color: #6c757d; font-size: 14px; margin: 0; }
    [data-testid="stFileUploader"], [data-testid="stCameraInput"] { animation: fadeIn 0.6s ease both; }
    [data-testid="stFileUploaderDropzone"] { border-radius: 16px !important; border: 2px dashed #f0a0a0 !important; transition: all 0.3s ease !important; }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: #dc2626 !important; background: #fff5f5 !important; }
    .result-anemia, .result-non-anemia { animation: popIn 0.5s ease both; }
    .logo-container { text-align: center; padding: 20px; background: transparent; border-radius: 30px; margin-bottom: 10px; }
    .logo-image { width: 100%; max-width: 500px; height: auto; object-fit: contain; margin: 0 auto; display: block; transition: transform 0.4s ease; filter: drop-shadow(0 15px 25px rgba(220,38,38,0.2)); }
    .logo-image:hover { transform: scale(1.02); }
    .app-title { text-align: center; font-size: 52px; font-weight: 800; background: linear-gradient(135deg, #8b0000, #dc2626, #ef4444); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 5px 0; letter-spacing: 2px; }
    .app-subtitle-ar { text-align: center; font-size: 28px; color: #b91c1c; margin: 5px 0; font-weight: 600; }
    .app-subtitle { text-align: center; font-size: 20px; color: #dc2626; margin: 5px 0; font-style: italic; }
    .app-tagline { text-align: center; font-size: 16px; color: #6c757d; margin-top: 5px; padding-bottom: 15px; border-bottom: 2px solid #dc2626; display: inline-block; width: auto; }
    .tagline-container { text-align: center; margin-bottom: 20px; }
    .model-card { background: white; border-radius: 20px; padding: 20px; text-align: center; transition: all 0.3s ease; border: 1px solid #ffcccc; box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
    .model-card:hover { transform: translateY(-5px); box-shadow: 0 10px 25px rgba(220,38,38,0.15); border-color: #dc2626; }
    .model-icon { font-size: 48px; margin-bottom: 15px; }
    .model-title { font-size: 22px; font-weight: 700; color: #dc2626; margin: 10px 0; }
    .model-desc { font-size: 14px; color: #6c757d; line-height: 1.5; }
    .model-badge { display: inline-block; background: linear-gradient(90deg, #dc2626, #ef4444); color: white; padding: 5px 15px; border-radius: 20px; font-size: 12px; margin-top: 10px; }
    .result-anemia { background: linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%); border-radius: 20px; padding: 25px; margin: 20px 0; border: 2px solid #dc2626; box-shadow: 0 10px 25px rgba(220,38,38,0.15); }
    .result-non-anemia { background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%); border-radius: 20px; padding: 25px; margin: 20px 0; border: 2px solid #16a34a; box-shadow: 0 10px 25px rgba(22,163,74,0.15); }
    .stButton > button { background: linear-gradient(90deg, #dc2626, #ef4444, #f87171); color: white; border: none; border-radius: 40px; padding: 14px 35px; font-size: 16px; font-weight: 600; transition: all 0.3s ease; width: 100%; box-shadow: 0 4px 10px rgba(220,38,38,0.3); }
    .stButton > button:hover { transform: scale(1.02); background: linear-gradient(90deg, #b91c1c, #dc2626, #ef4444); box-shadow: 0 6px 15px rgba(220,38,38,0.4); }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #dc2626, #ef4444); border-radius: 10px; }
    .disclaimer { background: linear-gradient(135deg, #fff5f5, #ffe0e0); border-radius: 15px; padding: 15px; text-align: center; font-size: 12px; color: #495057; margin-top: 30px; border-left: 4px solid #dc2626; }
    .stImage { border-radius: 16px; overflow: hidden; box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
    .stRadio > div { gap: 30px; justify-content: center; }
    .stRadio label { font-size: 16px; font-weight: 500; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #dc2626; }
    .section-title { font-size: 28px; font-weight: 700; color: #dc2626; margin: 20px 0 15px 0; padding-bottom: 10px; border-bottom: 3px solid #dc2626; display: inline-block; }
    .section-container { text-align: center; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ========== JavaScript للكاميرا ==========
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

# ========== الشعار ==========
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

st.markdown("""
<div class="app-title fade-in-up">AnemicCheck</div>
<div class="app-subtitle-ar fade-in-up">رؤية ميد</div>
<div class="app-subtitle fade-in-up">Détection d'Anémie par Intelligence Artificielle</div>
<div class="tagline-container fade-in-up">
    <div class="app-tagline">فحص فقر الدم غير الجراحي • Dépistage non invasif</div>
</div>
""", unsafe_allow_html=True)

# ========== عرض النماذج ==========
st.markdown('<div class="section-container"><span class="section-title">🧠 Modèles d\'IA utilisés</span></div>', unsafe_allow_html=True)
col_model1, col_model2 = st.columns(2)
with col_model1:
    st.markdown("""
    <div class="model-card fade-in-up">
        <div class="model-icon">🖼️🔬</div>
        <div class="model-title">U-Net</div>
        <div class="model-desc"><b>Segmentation</b><br>Encoder: ResNet34<br>Extraction du contour / الماسك الخام<br><span class="model-badge">Segmentation</span></div>
    </div>
    """, unsafe_allow_html=True)
with col_model2:
    st.markdown("""
    <div class="model-card fade-in-up" style="animation-delay: 0.1s;">
        <div class="model-icon">🧠📊</div>
        <div class="model-title">EfficientNet-B3</div>
        <div class="model-desc"><b>Classification</b><br>Architecture EfficientNet<br>Détection d'anémie<br><span class="model-badge">Classification</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(220,38,38,0.05); border-radius: 15px; padding: 15px; margin: 10px 0; text-align: center;">
    <p style="color: #666; margin: 0;">
        <b>📊 Pipeline:</b> Image → U-Net (Masque) → Extraction par Crop (Bounding Box) → EfficientNet-B3 (Classification)
    </p>
</div>
""", unsafe_allow_html=True)

# ========== رفع الصورة ==========
st.markdown("---")
st.markdown('<div class="section-container fade-in-up"><span class="section-title">📸 Méthode d\'acquisition</span></div>', unsafe_allow_html=True)

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

# =============================================================
# ===== الدوال الجديدة (تعتمد على القص فقط، بدون إضافات) =====
# =============================================================
def extract_eyelid_region(img, mask, padding=15):
    """
    تستقبل الصورة الأصلية والماسك الخام من U-Net.
    تقوم بإيجاد أكبر منطقة في الماسك، ثم تقص الصورة الأصلية على حسب هاد المنطقة.
    بدون أي معالجة إضافية (لا تنعيم، ولا مورفولوجيا).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        if w > 0 and h > 0:
            cropped_img = img[y:y+h, x:x+w]
            cropped_mask = mask[y:y+h, x:x+w]
            return cropped_img, cropped_mask, (x, y, w, h)
    
    # إذا لم يتم العثور على كونتور، نرجع الصورة كاملة
    return img, mask, None

# ===== دالة التصنيف (بدون قلب، مع تطبيع) =====
def predict_anemia(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        raw_pred = torch.sigmoid(output).item()
    
    # استخدام القيمة الخام مباشرة (بدون قلب)
    corrected_pred = raw_pred
    
    if corrected_pred >= 0.5:
        result = "Anemic"
        confidence = corrected_pred * 100
    else:
        result = "Non Anemic"
        confidence = (1 - corrected_pred) * 100
    
    return result, confidence, corrected_pred, raw_pred

# ========== المعالجة ==========
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
        # قراءة الصورة
        img = np.array(Image.open(uploaded).convert('RGB'))
        
        # ========== 1. تمرير الصورة إلى U-Net ==========
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
        
        # ========== 2. استخراج المنطقة بالقص (Crop) فقط ==========
        # نستدعي الدالة الجديدة التي تعتمد فقط على القص، بدون clean_mask
        cropped_img, cropped_mask, bbox = extract_eyelid_region(img, raw_mask, padding=15)
        
        # ========== 3. تصنيف المنطقة المقصوصة ==========
        result, confidence, corrected_pred, raw_pred = predict_anemia(classifier_model, cropped_img, classifier_device)
        
        # حساب النسب
        anemia_percent = corrected_pred * 100
        non_anemia_percent = (1 - corrected_pred) * 100
        
        st.success("✅ Analyse terminée avec succès")

        # ========== عرض النتائج ==========
        st.markdown("## 📊 Résultats de l'analyse")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🖼️ Image originale**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("**🎭 Masque brut (U-Net)**")
            st.image(raw_mask, use_container_width=True, clamp=True)
        with col3:
            st.markdown("**✂️ Région extraite (Crop)**")
            st.image(cropped_img, use_container_width=True)
        
        # إحصائيات بسيطة
        area = np.sum(raw_mask > 0) / 255
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Surface du masque", f"{area:.0f} px²")
        if bbox:
            with col_stat2:
                st.metric("Bounding Box", f"{bbox[2]}x{bbox[3]} px")
        
        # ========== التشخيص ==========
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
        
        # ========== المخطط ==========
        st.markdown("### 📈 Niveau d'anémie")
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ['Non anemic', 'Anemic']
        values = [non_anemia_percent, anemia_percent]
        colors = ['#10b981', '#dc2626']
        bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='white', linewidth=2)
        ax.set_ylim([0, 100])
        ax.set_ylabel('Pourcentage (%)', fontsize=12)
        ax.set_title('Probabilité d\'anémie (basée sur la région extraite)', fontsize=14, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{val:.1f}%', ha='center', fontweight='bold', fontsize=14, 
                    color='#8b0000' if val == anemia_percent else '#166534')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        
        # ========== تفاصيل تقنية ==========
        with st.expander("📈 Détails techniques"):
            st.write(f"**Diagnostic:** {result}")
            st.write(f"**Confiance:** {confidence:.2f}%")
            st.write(f"**Valeur sigmoïde brute:** {raw_pred:.4f}")
            st.write(f"**Valeur utilisée:** {corrected_pred:.4f} (aucune correction appliquée)")
            st.write(f"**Règle:** Si >= 0.5 → Anémie")
            st.write(f"**Méthode d'extraction:** Crop (Bounding Box) sans nettoyage ni morphologie")
            st.write(f"**Appareil:** {'GPU' if classifier_device.type == 'cuda' else 'CPU'}")
        
        # ========== تنويه ==========
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <b>Avertissement médical / تنويه طبي</b><br>
            Ce résultat est fourni à titre indicatif et ne remplace pas un avis médical professionnel.
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
    
    with st.expander("ℹ️ Guide d'utilisation"):
        st.markdown("""
        **Comment fonctionne l'application ?**
        1. **U-Net** : Génère un masque (segmentation) pour localiser la zone.
        2. **Extraction par Crop** : La zone est extraite par simple découpage (Bounding Box) sans modifications.
        3. **EfficientNet-B3** : Analyse la zone extraite.
        """)
