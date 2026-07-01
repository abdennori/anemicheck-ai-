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
from model_loader import load_unet_model, load_classifier_model

# ========== إعداد الصفحة ==========
st.set_page_config(
    page_title="AnemiCheck - Détection d'Anémie",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== CSS (تصميم طبي احترافي) ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #f8fafc; }
    
    .header {
        background: white;
        padding: 1rem 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header h1 { font-size: 28px; font-weight: 800; color: #0f172a; margin: 0; }
    .header h1 span { color: #e11d48; }
    .header .badge {
        background: #e11d48;
        color: white;
        padding: 6px 16px;
        border-radius: 40px;
        font-size: 14px;
        font-weight: 600;
    }
    
    .upload-card {
        background: white;
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        text-align: center;
        border: 2px dashed #cbd5e1;
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    .upload-card:hover {
        border-color: #e11d48;
        background: #fef2f2;
    }
    .upload-card .icon { font-size: 48px; }
    .upload-card h3 { font-weight: 600; color: #0f172a; margin: 12px 0 6px; }
    .upload-card p { color: #64748b; font-size: 14px; }
    
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        margin-top: 1.5rem;
        border-left: 6px solid #e11d48;
    }
    .result-card.positive { border-left-color: #dc2626; background: #fef2f2; }
    .result-card.negative { border-left-color: #16a34a; background: #f0fdf4; }
    .result-card h2 { font-size: 26px; font-weight: 700; margin: 0 0 6px; }
    .result-card .confidence { font-size: 18px; font-weight: 600; color: #1e293b; }
    .result-card .sub { font-size: 14px; color: #64748b; }
    
    .chart-container {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        margin-top: 1.5rem;
    }
    
    .disclaimer {
        background: #fefce8;
        border-radius: 16px;
        padding: 1rem 1.5rem;
        border-left: 6px solid #f59e0b;
        margin-top: 2rem;
        font-size: 13px;
        color: #4b5563;
        line-height: 1.6;
    }
    .disclaimer strong { color: #b45309; }
    
    #MainMenu, footer, .stDeployButton { display: none; }
    
    .stButton > button {
        background: #e11d48;
        color: white;
        border: none;
        border-radius: 40px;
        padding: 12px 28px;
        font-weight: 600;
        transition: 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background: #be123c;
        transform: scale(1.02);
        box-shadow: 0 8px 24px rgba(225,29,72,0.25);
    }
    
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        margin: 2rem 0 1rem;
        padding-bottom: 8px;
        border-bottom: 3px solid #e11d48;
        display: inline-block;
    }
    
    .stImage img { border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
    
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    [data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
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
            <img src="data:image/png;base64,{logo}" style="height:50px;">
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

# ========== UPLOAD ZONE ==========
st.markdown("""
<div class="upload-card">
    <div class="icon">📸</div>
    <h3>Téléchargez une image de l'œil</h3>
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
    """Nettoyer le masque : garder les gros contours, fermeture/ouverture morphologique"""
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
    """Amélioration de la conjonctive avec CLAHE + flou + netteté"""
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
    """
    Extrait la région de la conjonctive à partir du masque.
    Retourne : image recadrée améliorée, masque final, coordonnées (optionnel)
    """
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
            # On retourne aussi le masque recadré pour visualisation si besoin
            mask_cropped = mask[y:y+h, x:x+w]
            return enhanced, mask_cropped, (x, y, w, h)
    # Fallback : appliquer le masque à toute l'image
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
        pred = torch.sigmoid(out).item()  # valeur brute entre 0 et 1
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

# ========== TRAITEMENT ==========
if uploaded is not None:
    with st.spinner("🔃 Chargement des modèles..."):
        unet_model, unet_device, clf_model, clf_device = load_models()

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

        # Extraction de la conjonctive (améliorée)
        conj_enhanced, final_mask, bbox = extract_best_conjunctiva(img, raw_mask)

        # Classification
        result, confidence, raw_pred = predict_anemia(clf_model, conj_enhanced, clf_device)

        # Pour le graphique
        anemia_pct = raw_pred * 100
        non_pct = (1 - raw_pred) * 100

        st.success("✅ Analyse terminée avec succès")

        # --- Affichage des images ---
        st.markdown('<div class="section-title">📊 Résultats de l\'analyse</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🖼️ Image originale**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("**🎭 Masque final (nettoyé)**")
            st.image(final_mask, use_container_width=True, clamp=True)
        with col3:
            st.markdown("**👁️ Conjonctive extraite (améliorée)**")
            # Ici on affiche UNIQUEMENT la conjonctive recadrée
            st.image(conj_enhanced, use_container_width=True)

        # --- Métriques ---
        before = np.sum(raw_mask > 0) / 255
        after = np.sum(final_mask > 0) / 255
        reduction = ((before - after) / before * 100) if before > 0 else 0

        m1, m2 = st.columns(2)
        with m1:
            st.metric("📐 Surface segmentée", f"{after:.0f} px²")
        with m2:
            st.metric("🧼 Nettoyage", f"{reduction:.1f}% d'artefacts")

        # --- Diagnostic ---
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

        # --- Graphique à barres ---
        st.markdown('<div class="section-title">📈 Distribution des probabilités</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,5))
        cats = ['Non Anemic', 'Anemic']
        vals = [non_pct, anemia_pct]
        colors = ['#10b981', '#dc2626']
        bars = ax.bar(cats, vals, color=colors, width=0.5, edgecolor='white', linewidth=2)
        ax.set_ylim(0,100)
        ax.set_ylabel('Pourcentage (%)')
        ax.set_title('Probabilité estimée par le modèle', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

        # --- Détails techniques ---
        with st.expander("📘 Détails techniques"):
            st.write(f"**Modèle de segmentation:** U‑Net (ResNet34)")
            st.write(f"**Modèle de classification:** EfficientNet‑B3")
            st.write(f"**Appareil:** {'GPU' if clf_device.type == 'cuda' else 'CPU'}")
            st.write(f"**Valeur sigmoïde brute:** {raw_pred:.4f}")
            st.write(f"**Probabilité Anémie:** {anemia_pct:.1f}%")
            st.write(f"**Probabilité Non Anémie:** {non_pct:.1f}%")
            st.write("**Prétraitement:** CLAHE + Filtrage + Netteté sur la conjonctive")
            st.write("**Décision:** 100% autonome, sans correction humaine")

        # --- Avertissement médical ---
        st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ Avertissement médical</strong><br>
            Ce résultat est généré par un modèle d'intelligence artificielle et ne remplace en aucun cas l'avis d'un médecin.<br>
            Consultez un professionnel de santé pour un diagnostic fiable.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:white; border-radius:24px; padding:3rem; text-align:center; box-shadow:0 4px 16px rgba(0,0,0,0.04); margin-top:2rem;">
        <div style="font-size:64px;">👁️</div>
        <h3 style="color:#0f172a;">Prêt à analyser</h3>
        <p style="color:#64748b;">Utilisez le formulaire ci-dessus pour charger une image de l'œil.</p>
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
