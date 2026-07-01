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
from model_loader import load_unet_model, load_classifier_model

# ========== إعداد الصفحة ==========
st.set_page_config(
    page_title="AnemiCheck - Détection d'Anémie",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS / تصميم متقدم مع أنيميشن ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* خلفية متدرجة متحركة */
    .stApp {
        background: linear-gradient(-45deg, #f8fafc, #eef2ff, #f1f5f9, #f8fafc);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* شريط علوي مع تأثير زجاجي */
    .header {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 0.8rem 2rem;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.06);
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid rgba(255,255,255,0.3);
        animation: slideDown 0.6s ease;
    }
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .header h1 {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .header h1 span { color: #e11d48; }
    .header .badge {
        background: linear-gradient(135deg, #e11d48, #be123c);
        color: white;
        padding: 6px 18px;
        border-radius: 40px;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(225,29,72,0.25);
    }
    
    /* بطاقة رفع الصورة - Glassmorphism */
    .upload-card {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 28px;
        padding: 2.5rem 2rem;
        text-align: center;
        border: 2px dashed rgba(225,29,72,0.2);
        transition: all 0.4s ease;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.04);
        animation: fadeUp 0.8s ease;
    }
    .upload-card:hover {
        border-color: #e11d48;
        background: rgba(255,255,255,0.8);
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(225,29,72,0.08);
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .upload-card .icon {
        font-size: 56px;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .upload-card h3 {
        font-weight: 700;
        color: #0f172a;
        margin: 12px 0 6px;
    }
    .upload-card p {
        color: #64748b;
        font-size: 14px;
    }
    
    /* بطاقات النتائج مع أنيميشن */
    .result-card {
        border-radius: 24px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        border-left: 6px solid #e11d48;
        animation: popIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    .result-card.positive {
        background: rgba(254, 242, 242, 0.8);
        border-left-color: #dc2626;
        backdrop-filter: blur(4px);
    }
    .result-card.negative {
        background: rgba(240, 253, 244, 0.8);
        border-left-color: #16a34a;
        backdrop-filter: blur(4px);
    }
    .result-card h2 {
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 6px;
    }
    .result-card .confidence {
        font-size: 18px;
        font-weight: 600;
        color: #1e293b;
    }
    .result-card .sub {
        font-size: 14px;
        color: #64748b;
    }
    
    /* شريط جانبي بتصميم زجاجي */
    .sidebar-glass {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
        animation: fadeUp 0.8s ease;
    }
    .sidebar-glass h4 {
        color: #0f172a;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sidebar-glass p, .sidebar-glass li {
        color: #334155;
        font-size: 14px;
        line-height: 1.6;
    }
    .sidebar-glass ul {
        padding-left: 1.2rem;
    }
    .sidebar-glass .qr-container {
        display: flex;
        justify-content: center;
        margin: 0.8rem 0;
    }
    .sidebar-glass .qr-container img {
        border-radius: 16px;
        border: 1px solid rgba(225,29,72,0.15);
        background: white;
        padding: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        transition: transform 0.3s ease;
    }
    .sidebar-glass .qr-container img:hover {
        transform: scale(1.05);
    }
    
    .doctor-btn {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 15px;
        cursor: not-allowed;
        opacity: 0.6;
        width: 100%;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(37,99,235,0.2);
        margin-top: 10px;
    }
    .doctor-btn:hover {
        opacity: 0.8;
        transform: scale(1.02);
    }
    .coming-badge {
        background: #f59e0b;
        color: white;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 12px;
        border-radius: 30px;
        margin-left: 8px;
        letter-spacing: 0.5px;
    }
    
    /* تنويه طبي */
    .disclaimer {
        background: rgba(254, 252, 232, 0.8);
        backdrop-filter: blur(4px);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        border-left: 6px solid #f59e0b;
        margin-top: 2rem;
        font-size: 13px;
        color: #4b5563;
        line-height: 1.6;
        animation: fadeUp 1s ease;
    }
    
    /* تحسينات عامة */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        margin: 2rem 0 1rem;
        padding-bottom: 8px;
        border-bottom: 3px solid #e11d48;
        display: inline-block;
        animation: fadeUp 0.6s ease;
    }
    .stImage img { border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
    
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(4px);
        border-radius: 16px;
        padding: 12px 16px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    #MainMenu, footer, .stDeployButton { display: none; }
    
    .stButton > button {
        background: linear-gradient(135deg, #e11d48, #be123c);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 12px 28px;
        font-weight: 600;
        transition: 0.3s;
        width: 100%;
        box-shadow: 0 4px 16px rgba(225,29,72,0.2);
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 24px rgba(225,29,72,0.3);
    }
    
    /* شريط التقدم المتحرك */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e11d48, #fb7185);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
def get_logo_base64():
    for name in ["logo.png", "logo.jpg", "logo.jpeg", "LOGO.png"]:
        if os.path.exists(name):
            with open(name, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None

logo = get_logo_base64()
if logo:
    st.markdown(f"""
    <div class="header">
        <div style="display:flex; align-items:center; gap:12px;">
            <img src="data:image/png;base64,{logo}" style="height:48px;">
            <h1>Anemi<span>Check</span></h1>
        </div>
        <div class="badge">🩺 IA Médicale</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="header">
        <h1>🩸 Anemi<span>Check</span></h1>
        <div class="badge">🩺 IA Médicale</div>
    </div>
    """, unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("---")
    
    # QR Code (sans lien texte)
    APP_URL = "https://hwaxrexkahkxaazwwjjr3d.streamlit.app/"
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&margin=10&color=e11d48&data={APP_URL}"
    
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>📱 Accès rapide</h4>
        <div class="qr-container">
            <img src="{qr_api_url}" width="160" height="160" alt="QR Code">
        </div>
        <p style="text-align:center; font-size:13px; color:#64748b;">Scannez pour ouvrir l'application</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Conseils médicaux
    st.markdown("""
    <div class="sidebar-glass">
        <h4>💡 Conseils santé</h4>
        <ul>
            <li><strong>Alimentation</strong> : privilégiez les aliments riches en fer (viande rouge, légumineuses, épinards).</li>
            <li><strong>Vitamine C</strong> : améliore l'absorption du fer (agrumes, kiwi, poivrons).</li>
            <li><strong>Hydratation</strong> : buvez au moins 1,5 L d'eau par jour.</li>
            <li><strong>Consultez</strong> un médecin en cas de fatigue persistante, pâleur ou essoufflement.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Consultation médecin (Coming Soon)
    st.markdown("""
    <div class="sidebar-glass">
        <h4>🩺 Consultation médecin <span class="coming-badge">Prochainement</span></h4>
        <p>Prenez rendez-vous ou discutez avec un professionnel de santé directement depuis l'application.</p>
        <div class="doctor-btn">📅 Réserver une téléconsultation</div>
        <p style="font-size:12px; color:#94a3b8; text-align:center; margin-top:8px;">🔜 Disponible dans une prochaine mise à jour</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("AnemiCheck v2.0 • IA médicale")

# ========== ZONE UPLOAD ==========
st.markdown("""
<div class="upload-card">
    <div class="icon">📸</div>
    <h3>Chargez une image de l'œil</h3>
    <p>Photo claire et bien éclairée de la conjonctive</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])
with col1:
    option = st.radio("Méthode", ["📁 Fichier", "📷 Caméra"], horizontal=True, label_visibility="collapsed")
with col2:
    if option == "📁 Fichier":
        uploaded = st.file_uploader("", type=["jpg","png","jpeg"], label_visibility="collapsed")
    else:
        uploaded = st.camera_input("", label_visibility="collapsed")

# ========== FONCTIONS DE TRAITEMENT AVANCÉ ==========
def clean_mask(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(cleaned, [c], -1, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned

def enhance_conjunctiva(image):
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enh = clahe.apply(l)
        l_enh = cv2.GaussianBlur(l_enh, (3,3), 0)
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        l_enh = cv2.filter2D(l_enh, -1, kernel)
        lab_enh = cv2.merge((l_enh, a, b))
        enhanced = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2RGB)
        return enhanced
    return image

def extract_best_conjunctiva(img, mask):
    mask = clean_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 15
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2*pad)
        h = min(img.shape[0] - y, h + 2*pad)
        if w > 0 and h > 0:
            cropped = img[y:y+h, x:x+w]
            enhanced = enhance_conjunctiva(cropped)
            mask_cropped = mask[y:y+h, x:x+w]
            return enhanced, mask_cropped, (x, y, w, h)
    conj = cv2.bitwise_and(img, img, mask=mask)
    enhanced = enhance_conjunctiva(conj)
    return enhanced, mask, None

# ========== CLASSIFICATION (SANS BIAIS) ==========
def predict_anemia(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        pred = torch.sigmoid(out).item()
    if pred >= 0.5:
        return "Anemic", pred * 100, pred
    else:
        return "Non Anemic", (1 - pred) * 100, pred

# ========== CHARGEMENT LAZY ==========
@st.cache_resource
def load_models():
    unet, dev_unet = load_unet_model()
    clf, dev_clf = load_classifier_model()
    return unet, dev_unet, clf, dev_clf

# ========== TRAITEMENT PRINCIPAL ==========
if uploaded is not None:
    with st.spinner("🔃 Chargement des modèles..."):
        unet_model, unet_device, clf_model, clf_device = load_models()

    # Barre de progression simulée pour l'effet
    progress_bar = st.progress(0)
    for i in range(10):
        time.sleep(0.05)
        progress_bar.progress((i+1)*10)

    with st.spinner("🔍 Analyse en cours..."):
        # Lecture et correction miroir
        img = np.array(Image.open(uploaded).convert('RGB'))
        img = cv2.flip(img, 1)

        # Segmentation U‑Net
        transform_unet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        tensor = transform_unet(img).unsqueeze(0).to(unet_device)
        with torch.no_grad():
            raw_mask = torch.sigmoid(unet_model(tensor)).squeeze().cpu().numpy()
            raw_mask = (raw_mask > 0.5).astype(np.uint8) * 255
            raw_mask = cv2.resize(raw_mask, (img.shape[1], img.shape[0]))

        # Extraction de la conjonctive
        conj_enhanced, final_mask, bbox = extract_best_conjunctiva(img, raw_mask)

        # Classification (sans aucune correction)
        result, confidence, raw_pred = predict_anemia(clf_model, conj_enhanced, clf_device)

        # Valeurs réelles
        anemia_pct = raw_pred * 100
        non_pct = (1 - raw_pred) * 100

        progress_bar.empty()
        st.success("✅ Analyse terminée avec succès")

        # --- IMAGES ---
        st.markdown('<div class="section-title">📊 Résultats de l\'analyse</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🖼️ Image originale**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("**🎭 Masque final**")
            st.image(final_mask, use_container_width=True, clamp=True)
        with col3:
            st.markdown("**👁️ Conjonctive optimisée**")
            st.image(conj_enhanced, use_container_width=True)

        # --- MÉTRIQUES ---
        before = np.sum(raw_mask > 0) / 255
        after = np.sum(final_mask > 0) / 255
        reduction = ((before - after) / before * 100) if before > 0 else 0

        m1, m2 = st.columns(2)
        with m1:
            st.metric("📐 Surface segmentée", f"{after:.0f} px²")
        with m2:
            st.metric("🧼 Nettoyage", f"{reduction:.1f}% d'artefacts")

        # --- DIAGNOSTIC ---
        st.markdown('<div class="section-title">🩺 Diagnostic</div>', unsafe_allow_html=True)
        col_res, col_conf = st.columns(2)
        with col_res:
            if result == "Anemic":
                st.markdown(f"""
                <div class="result-card positive">
                    <h2>🩸 Anémie</h2>
                    <div class="confidence">Probabilité : <strong>{anemia_pct:.1f}%</strong></div>
                    <div class="sub">Le modèle détecte une anémie probable.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card negative">
                    <h2>✅ Non Anémie</h2>
                    <div class="confidence">Probabilité : <strong>{non_pct:.1f}%</strong></div>
                    <div class="sub">Le modèle ne détecte pas d'anémie.</div>
                </div>
                """, unsafe_allow_html=True)
        with col_conf:
            st.metric("📊 Niveau de confiance", f"{confidence:.1f}%")
            st.progress(int(confidence))

        # --- GRAPHIQUE ---
        st.markdown('<div class="section-title">📈 Distribution des probabilités</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,5))
        cats = ['Non Anemic', 'Anemic']
        vals = [non_pct, anemia_pct]
        colors = ['#10b981', '#dc2626']
        bars = ax.bar(cats, vals, color=colors, width=0.5, edgecolor='white', linewidth=2)
        ax.set_ylim(0,100)
        ax.set_ylabel('Pourcentage (%)')
        ax.set_title('Probabilité estimée par le modèle (décision brute)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

        # --- DÉTAILS TECHNIQUES ---
        with st.expander("📘 Détails techniques"):
            st.write(f"**Modèle de segmentation:** U‑Net (ResNet34)")
            st.write(f"**Modèle de classification:** EfficientNet‑B3")
            st.write(f"**Appareil:** {'GPU' if clf_device.type == 'cuda' else 'CPU'}")
            st.write(f"**Valeur sigmoïde brute:** {raw_pred:.4f}")
            st.write(f"**Probabilité Anémie:** {anemia_pct:.1f}%")
            st.write(f"**Probabilité Non Anémie:** {non_pct:.1f}%")
            st.write("**Prétraitement:** CLAHE + Filtrage + Netteté")
            st.write("**Décision:** 100% autonome, sans biais")

        # --- AVERTISSEMENT ---
        st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ Avertissement médical</strong><br>
            Ce résultat est généré par un modèle d'intelligence artificielle et ne remplace en aucun cas l'avis d'un médecin.<br>
            Consultez un professionnel de santé pour un diagnostic fiable.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.6); backdrop-filter:blur(8px); border-radius:28px; padding:3rem; text-align:center; box-shadow:0 8px 32px rgba(0,0,0,0.04); margin-top:2rem; border:1px solid rgba(255,255,255,0.3); animation:fadeUp 0.8s ease;">
        <div style="font-size:64px; animation:float 3s ease-in-out infinite;">👁️</div>
        <h3 style="color:#0f172a;">Prêt à analyser</h3>
        <p style="color:#64748b;">Chargez une image de l'œil pour commencer le diagnostic.</p>
    </div>
    """, unsafe_allow_html=True)

# ========== GUIDE ==========
with st.expander("ℹ️ Comment ça marche ?"):
    st.markdown("""
    **1. Segmentation U‑Net**  
    Le modèle extrait automatiquement la conjonctive à partir de l'image de l'œil.

    **2. Prétraitement avancé**  
    La zone extraite est améliorée (CLAHE, filtrage, netteté) pour optimiser la classification.

    **3. Classification EfficientNet‑B3**  
    Le modèle analyse la conjonctive et calcule la probabilité d'anémie.

    **4. Résultat interprétable**  
    Vous obtenez un diagnostic clair avec un niveau de confiance.
    """)
