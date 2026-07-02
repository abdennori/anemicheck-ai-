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

# ========== إعداد الصفحة ==========
st.set_page_config(
    page_title="AnemiCheck - Détection d'Anémie",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== الترجمة ==========
LANGUAGES = {
    "ar": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "الذكاء الاصطناعي لصحة الدم",
        "badge_private": "أمين وموثوق",
        "badge_private_desc": "100% خصوصيتك محمية",
        "badge_available": "خدمة متاحة",
        "badge_available_desc": "24/7 نحن هنا من أجلك دائماً",
        "hero_title": "طبيبك معك بأقل من دقيقتين",
        "hero_desc": "تحليل ذكي لصورة العين لكشف الأنيميا بدقة وسرعة عالية",
        "hero_badge": "100% نتائج دقيقة وآمنة",
        "feature_1_title": "نتيجة سريعة",
        "feature_1_desc": "احصل على النتيجة فوراً مع تقرير مفصل",
        "feature_2_title": "رفع الصورة",
        "feature_2_desc": "قم برفع صورة واضحة للعين (JPG, PNG)",
        "feature_3_title": "موثوق ومعتمد",
        "feature_3_desc": "هذا التطبيق يساعد في الكشف المبكر عن الأنيميا",
        "feature_4_title": "أمان",
        "feature_4_desc": "بياناتك محمية بالكامل",
        "how_title": "كيف يعمل التطبيق؟",
        "how_step1": "رفع الصورة",
        "how_step1_desc": "ارفع صورة واضحة للعين",
        "how_step2": "تحليل ذكي",
        "how_step2_desc": "يستخدم النموذج الذكاء الاصطناعي للتحليل",
        "how_step3": "النتيجة الفورية",
        "how_step3_desc": "احصل على التشخيص في ثوانٍ",
        "trust_1": "أمان",
        "trust_2": "رعاية",
        "trust_3": "ثقة عالية",
        "version": "الإصدار 1.0.0",
        "accuracy": "دقة تصل إلى 96%",
        "fast": "سريع",
        "fast_desc": "النتيجة في أقل من دقيقتين",
        "smart": "ذكي",
        "smart_desc": "تقنية ذكاء اصطناعي متقدمة",
        "secure": "أمان",
        "secure_desc": "بياناتك محمية بالكامل",
        "upload_title": "رفع صورة العين",
        "upload_desc": "صورة واضحة للعين للكشف عن الأنيميا",
        "upload_method_file": "📁 رفع صورة",
        "upload_method_camera": "📷 تصوير",
        "upload_file_label": "اختر صورة من جهازك",
        "upload_camera_label": "التقط صورة بالكاميرا",
        "preview_title": "📷 معاينة الصورة",
        "preview_caption": "الصورة المرفوعة",
        "analyze_btn": "🔍 تحليل الصورة",
        "loading_models": "⏳ جارٍ تحميل النماذج...",
        "analyzing": "⏳ جارٍ التحليل...",
        "analysis_done": "✅ تم التحليل بنجاح",
        "results_title": "📊 نتائج التحليل",
        "result_original": "🖼️ الصورة الأصلية",
        "result_mask": "🎭 القناع النهائي",
        "result_conjunctiva": "👁️ الملتحمة المحسّنة",
        "metric_surface": "📐 المساحة المقطعة",
        "metric_cleaning": "🧼 التنظيف",
        "diagnostic_title": "🩺 التشخيص",
        "diagnostic_anemic": "🩸 فقر دم",
        "diagnostic_anemic_desc": "يوجد فقر دم",
        "diagnostic_non_anemic": "✅ لا يوجد فقر دم",
        "diagnostic_non_anemic_desc": "لا يوجد فقر دم",
        "diagnostic_confidence": "📊 مستوى الثقة",
        "chart_title": "📈 توزيع الاحتمالات",
        "chart_non": "غير مصاب",
        "chart_anemic": "مصاب",
        "history_title": "📋 سجل التحليلات",
        "history_date": "التاريخ",
        "history_diagnostic": "التشخيص",
        "history_confidence": "الثقة",
        "history_prob": "احتمال فقر الدم",
        "tech_details": "📘 التفاصيل التقنية",
        "tech_model_seg": "نموذج التجزئة",
        "tech_model_clf": "نموذج التصنيف",
        "tech_device": "الجهاز المستخدم",
        "tech_sigmoid": "قيمة السيجمويد الخام",
        "tech_prob_anemic": "احتمال فقر الدم",
        "tech_prob_non": "احتمال عدم الإصابة",
        "tech_preprocess": "المعالجة المسبقة",
        "tech_decision": "القرار",
        "tech_decision_value": "مستقل تماماً، بدون تحيز",
        "disclaimer": "⚠️ تنويه طبي",
        "disclaimer_text": "هذه النتيجة صادرة عن نموذج ذكاء اصطناعي ولا تغني عن استشارة الطبيب المختص.",
        "disclaimer_consult": "استشر طبيباً للحصول على تشخيص دقيق.",
        "welcome_title": "👁️ جاهز للتحليل",
        "welcome_desc": "حمّل صورة العين للبدء.",
        "lang_selector": "🌐 اللغة",
        "sidebar_qr_title": "📱 وصول سريع",
        "sidebar_qr_desc": "امسح الكود لفتح التطبيق",
        "sidebar_health_title": "💡 نصائح صحية",
        "sidebar_health_1": "التغذية: ركز على الأطعمة الغنية بالحديد.",
        "sidebar_health_2": "فيتامين C: يحسن امتصاص الحديد.",
        "sidebar_health_3": "الترطيب: اشرب 1.5 لتر ماء يومياً.",
        "sidebar_health_4": "استشر الطبيب عند الشعور بالتعب الشديد.",
        "sidebar_doctor_title": "🩺 استشارة طبيب",
        "sidebar_doctor_desc": "احجز موعداً أو تحدث مع طبيب مباشرة.",
        "sidebar_doctor_btn": "📅 حجز استشارة",
        "sidebar_doctor_soon": "🔜 متاح في التحديث القادم",
        "sidebar_version": "AnemiCheck v2.0 • الذكاء الاصطناعي الطبي",
        "nav_home": "🏠 الرئيسية",
        "nav_analyze": "🔬 تحليل صورة الدم",
        "nav_history": "📊 النتائج السابقة",
        "nav_info": "ℹ️ معلومات المرض",
        "nav_health": "💡 نصائح صحية",
        "nav_about": "📱 حول التطبيق",
    },
    "fr": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "L'IA pour la santé sanguine",
        "badge_private": "Sécurisé et fiable",
        "badge_private_desc": "100% de confidentialité garantie",
        "badge_available": "Service disponible",
        "badge_available_desc": "24/7 nous sommes là pour vous",
        "hero_title": "Votre médecin en moins de deux minutes",
        "hero_desc": "Analyse intelligente de l'image oculaire pour détecter l'anémie avec précision et rapidité",
        "hero_badge": "100% de résultats précis et sécurisés",
        "feature_1_title": "Résultat rapide",
        "feature_1_desc": "Obtenez le résultat immédiatement avec un rapport détaillé",
        "feature_2_title": "Téléchargement d'image",
        "feature_2_desc": "Téléchargez une image claire de l'œil (JPG, PNG)",
        "feature_3_title": "Fiable et approuvé",
        "feature_3_desc": "Cette application aide à la détection précoce de l'anémie",
        "feature_4_title": "Sécurité",
        "feature_4_desc": "Vos données sont entièrement protégées",
        "how_title": "Comment ça marche ?",
        "how_step1": "Téléchargement",
        "how_step1_desc": "Téléchargez une image claire de l'œil",
        "how_step2": "Analyse intelligente",
        "how_step2_desc": "Le modèle d'IA analyse l'image",
        "how_step3": "Résultat instantané",
        "how_step3_desc": "Obtenez le diagnostic en quelques secondes",
        "trust_1": "Sécurité",
        "trust_2": "Soins",
        "trust_3": "Haute confiance",
        "version": "Version 1.0.0",
        "accuracy": "Précision jusqu'à 96%",
        "fast": "Rapide",
        "fast_desc": "Résultat en moins de 2 minutes",
        "smart": "Intelligent",
        "smart_desc": "Technologie IA avancée",
        "secure": "Sécurisé",
        "secure_desc": "Vos données sont protégées",
        "upload_title": "Télécharger une image de l'œil",
        "upload_desc": "Image claire de l'œil pour la détection de l'anémie",
        "upload_method_file": "📁 Télécharger",
        "upload_method_camera": "📷 Appareil photo",
        "upload_file_label": "Choisissez une image depuis votre appareil",
        "upload_camera_label": "Prenez une photo avec la caméra",
        "preview_title": "📷 Aperçu de l'image",
        "preview_caption": "Image chargée",
        "analyze_btn": "🔍 Analyser l'image",
        "loading_models": "⏳ Chargement des modèles...",
        "analyzing": "⏳ Analyse en cours...",
        "analysis_done": "✅ Analyse terminée",
        "results_title": "📊 Résultats de l'analyse",
        "result_original": "🖼️ Image originale",
        "result_mask": "🎭 Masque final",
        "result_conjunctiva": "👁️ Conjonctive optimisée",
        "metric_surface": "📐 Surface segmentée",
        "metric_cleaning": "🧼 Nettoyage",
        "diagnostic_title": "🩺 Diagnostic",
        "diagnostic_anemic": "🩸 Anémie",
        "diagnostic_anemic_desc": "Anémie détectée",
        "diagnostic_non_anemic": "✅ Non Anémie",
        "diagnostic_non_anemic_desc": "Pas d'anémie détectée",
        "diagnostic_confidence": "📊 Niveau de confiance",
        "chart_title": "📈 Distribution des probabilités",
        "chart_non": "Non Anémique",
        "chart_anemic": "Anémique",
        "history_title": "📋 Historique des analyses",
        "history_date": "Date",
        "history_diagnostic": "Diagnostic",
        "history_confidence": "Confiance",
        "history_prob": "Probabilité Anémie",
        "tech_details": "📘 Détails techniques",
        "tech_model_seg": "Modèle de segmentation",
        "tech_model_clf": "Modèle de classification",
        "tech_device": "Appareil utilisé",
        "tech_sigmoid": "Valeur sigmoïde brute",
        "tech_prob_anemic": "Probabilité Anémie",
        "tech_prob_non": "Probabilité Non Anémie",
        "tech_preprocess": "Prétraitement",
        "tech_decision": "Décision",
        "tech_decision_value": "100% autonome, sans biais",
        "disclaimer": "⚠️ Avertissement médical",
        "disclaimer_text": "Ce résultat est généré par un modèle d'IA et ne remplace pas un avis médical.",
        "disclaimer_consult": "Consultez un professionnel de santé.",
        "welcome_title": "👁️ Prêt à analyser",
        "welcome_desc": "Téléchargez une image de l'œil pour commencer.",
        "lang_selector": "🌐 Langue",
        "sidebar_qr_title": "📱 Accès rapide",
        "sidebar_qr_desc": "Scannez pour ouvrir l'application",
        "sidebar_health_title": "💡 Conseils santé",
        "sidebar_health_1": "Alimentation : aliments riches en fer.",
        "sidebar_health_2": "Vitamine C : améliore l'absorption du fer.",
        "sidebar_health_3": "Hydratation : 1,5 L d'eau par jour.",
        "sidebar_health_4": "Consultez un médecin en cas de fatigue.",
        "sidebar_doctor_title": "🩺 Consultation médecin",
        "sidebar_doctor_desc": "Prenez rendez-vous ou discutez avec un médecin.",
        "sidebar_doctor_btn": "📅 Réserver",
        "sidebar_doctor_soon": "🔜 Bientôt disponible",
        "sidebar_version": "AnemiCheck v2.0 • IA médicale",
        "nav_home": "🏠 Accueil",
        "nav_analyze": "🔬 Analyser l'image",
        "nav_history": "📊 Historique",
        "nav_info": "ℹ️ Infos maladie",
        "nav_health": "💡 Conseils santé",
        "nav_about": "📱 À propos",
    },
    "en": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "AI for Blood Health",
        "badge_private": "Secure & Trusted",
        "badge_private_desc": "100% privacy protected",
        "badge_available": "Service Available",
        "badge_available_desc": "24/7 we are here for you",
        "hero_title": "Your doctor in less than two minutes",
        "hero_desc": "Intelligent analysis of eye images to detect anemia with high accuracy and speed",
        "hero_badge": "100% accurate and secure results",
        "feature_1_title": "Quick Result",
        "feature_1_desc": "Get instant results with a detailed report",
        "feature_2_title": "Upload Image",
        "feature_2_desc": "Upload a clear eye image (JPG, PNG)",
        "feature_3_title": "Reliable & Approved",
        "feature_3_desc": "This app helps in early detection of anemia",
        "feature_4_title": "Security",
        "feature_4_desc": "Your data is fully protected",
        "how_title": "How it works?",
        "how_step1": "Upload",
        "how_step1_desc": "Upload a clear eye image",
        "how_step2": "Smart Analysis",
        "how_step2_desc": "AI model analyzes the image",
        "how_step3": "Instant Result",
        "how_step3_desc": "Get diagnosis in seconds",
        "trust_1": "Security",
        "trust_2": "Care",
        "trust_3": "High Trust",
        "version": "Version 1.0.0",
        "accuracy": "Accuracy up to 96%",
        "fast": "Fast",
        "fast_desc": "Result in less than 2 minutes",
        "smart": "Smart",
        "smart_desc": "Advanced AI technology",
        "secure": "Secure",
        "secure_desc": "Your data is protected",
        "upload_title": "Upload Eye Image",
        "upload_desc": "Clear eye image for anemia detection",
        "upload_method_file": "📁 Upload",
        "upload_method_camera": "📷 Camera",
        "upload_file_label": "Choose an image from your device",
        "upload_camera_label": "Take a photo with camera",
        "preview_title": "📷 Image Preview",
        "preview_caption": "Uploaded image",
        "analyze_btn": "🔍 Analyze Image",
        "loading_models": "⏳ Loading models...",
        "analyzing": "⏳ Analyzing...",
        "analysis_done": "✅ Analysis complete",
        "results_title": "📊 Analysis Results",
        "result_original": "🖼️ Original Image",
        "result_mask": "🎭 Final Mask",
        "result_conjunctiva": "👁️ Enhanced Conjunctiva",
        "metric_surface": "📐 Segmented Area",
        "metric_cleaning": "🧼 Cleaning",
        "diagnostic_title": "🩺 Diagnosis",
        "diagnostic_anemic": "🩸 Anemia",
        "diagnostic_anemic_desc": "Anemia detected",
        "diagnostic_non_anemic": "✅ No Anemia",
        "diagnostic_non_anemic_desc": "No anemia detected",
        "diagnostic_confidence": "📊 Confidence Level",
        "chart_title": "📈 Probability Distribution",
        "chart_non": "Non Anemic",
        "chart_anemic": "Anemic",
        "history_title": "📋 Analysis History",
        "history_date": "Date",
        "history_diagnostic": "Diagnosis",
        "history_confidence": "Confidence",
        "history_prob": "Anemia Probability",
        "tech_details": "📘 Technical Details",
        "tech_model_seg": "Segmentation Model",
        "tech_model_clf": "Classification Model",
        "tech_device": "Device Used",
        "tech_sigmoid": "Raw Sigmoid Value",
        "tech_prob_anemic": "Anemia Probability",
        "tech_prob_non": "Non-Anemia Probability",
        "tech_preprocess": "Preprocessing",
        "tech_decision": "Decision",
        "tech_decision_value": "100% autonomous, no bias",
        "disclaimer": "⚠️ Medical Disclaimer",
        "disclaimer_text": "This result is generated by an AI model and does not replace professional medical advice.",
        "disclaimer_consult": "Consult a healthcare professional.",
        "welcome_title": "👁️ Ready to analyze",
        "welcome_desc": "Upload an eye image to start.",
        "lang_selector": "🌐 Language",
        "sidebar_qr_title": "📱 Quick Access",
        "sidebar_qr_desc": "Scan to open the app",
        "sidebar_health_title": "💡 Health Tips",
        "sidebar_health_1": "Diet: focus on iron-rich foods.",
        "sidebar_health_2": "Vitamin C: improves iron absorption.",
        "sidebar_health_3": "Hydration: drink 1.5 L of water daily.",
        "sidebar_health_4": "Consult a doctor if you experience fatigue.",
        "sidebar_doctor_title": "🩺 Doctor Consultation",
        "sidebar_doctor_desc": "Book an appointment or chat with a doctor.",
        "sidebar_doctor_btn": "📅 Book Appointment",
        "sidebar_doctor_soon": "🔜 Coming soon",
        "sidebar_version": "AnemiCheck v2.0 • Medical AI",
        "nav_home": "🏠 Home",
        "nav_analyze": "🔬 Analyze Image",
        "nav_history": "📊 History",
        "nav_info": "ℹ️ Disease Info",
        "nav_health": "💡 Health Tips",
        "nav_about": "📱 About",
    }
}

# ========== دالة الترجمة ==========
def t(key):
    lang = st.session_state.get("language", "fr")
    return LANGUAGES.get(lang, LANGUAGES["fr"]).get(key, key)

# ========== CSS ==========
st.markdown("""
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=Tajawal:wght@400;500;700;800&display=swap');

    :root {
        --ink: #0B1220;
        --slate: #47546B;
        --slate-soft: #7C8AA5;
        --line: #E6EAF2;
        --surface: #FFFFFF;
        --bg: #F4F7FC;
        --primary-900: #0B2E86;
        --primary-700: #1450D6;
        --primary-600: #2563EB;
        --primary-400: #5B8DEF;
        --primary-100: #E8EFFE;
        --teal: #0EA5A4;
        --teal-100: #E3FBF8;
        --success: #10A85A;
        --success-100: #E7F9EF;
        --danger: #E23F4E;
        --danger-100: #FDECEE;
        --amber: #DE9A1F;
        --amber-100: #FDF3DF;
        --r-sm: 12px;
        --r-md: 18px;
        --r-lg: 26px;
        --r-xl: 32px;
        --shadow-sm: 0 2px 8px rgba(11,18,32,0.04);
        --shadow-md: 0 10px 30px rgba(11,18,32,0.07);
        --shadow-lg: 0 24px 60px rgba(11,46,134,0.14);
    }

    html, body, [class*="css"] { font-family: 'Inter', 'Tajawal', sans-serif; }
    h1, h2, h3, h4, h5, .brand-font { font-family: 'Plus Jakarta Sans', 'Tajawal', sans-serif; }

    .stApp {
        background:
            radial-gradient(1100px 480px at 85% -10%, rgba(37,99,235,0.08), transparent 60%),
            radial-gradient(900px 420px at -10% 10%, rgba(14,165,164,0.06), transparent 55%),
            var(--bg);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1B3F 0%, #0B2E86 100%);
        border-right: none;
    }
    section[data-testid="stSidebar"] * { color: #E8EFFE; }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.08);
        border-radius: var(--r-sm);
        border: 1px solid rgba(255,255,255,0.16);
    }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.14); margin: 1.1rem 0; }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(18px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes floatY {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    @keyframes dashMove {
        to { stroke-dashoffset: -400; }
    }
    @keyframes pulseRing {
        0% { box-shadow: 0 0 0 0 rgba(255,255,255,0.35); }
        100% { box-shadow: 0 0 0 14px rgba(255,255,255,0); }
    }

    /* ===== TOP NAVBAR ===== */
    .navbar {
        background: rgba(255,255,255,0.78);
        backdrop-filter: blur(18px) saturate(160%);
        -webkit-backdrop-filter: blur(18px) saturate(160%);
        border: 1px solid rgba(255,255,255,0.6);
        border-radius: var(--r-lg);
        padding: 0.9rem 1.6rem;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        box-shadow: var(--shadow-sm);
        animation: fadeUp 0.5s ease;
    }
    .navbar-brand { display: flex; align-items: center; gap: 12px; }
    .navbar-brand .mark {
        width: 46px; height: 46px; border-radius: 14px;
        background: linear-gradient(135deg, var(--primary-600), var(--primary-900));
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; box-shadow: var(--shadow-sm);
        flex-shrink: 0;
    }
    .navbar-brand img.mark-img { width: 46px; height: 46px; border-radius: 14px; object-fit: cover; box-shadow: var(--shadow-sm); }
    .navbar-brand h1 { font-size: 21px; font-weight: 800; color: var(--ink); margin: 0; letter-spacing: -0.01em; }
    .navbar-brand .subtitle { font-size: 12.5px; color: var(--slate-soft); font-weight: 500; margin-top: 1px; }
    .navbar-badges { display: flex; gap: 10px; flex-wrap: wrap; }
    .navbar-chip {
        display: flex; align-items: center; gap: 9px;
        background: var(--primary-100);
        border: 1px solid rgba(37,99,235,0.12);
        padding: 7px 14px 7px 10px;
        border-radius: 30px;
    }
    .navbar-chip .chip-ic {
        width: 26px; height: 26px; border-radius: 50%;
        background: var(--surface); display:flex; align-items:center; justify-content:center;
        font-size: 13px; box-shadow: var(--shadow-sm);
    }
    .navbar-chip .chip-text { line-height: 1.25; text-align: left; }
    .navbar-chip .chip-title { font-size: 12.5px; font-weight: 700; color: var(--primary-900); display: block; }
    .navbar-chip .chip-sub { font-size: 11px; color: var(--slate-soft); display: block; }

    /* ===== SIDEBAR PANELS ===== */
    .sidebar-glass {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: var(--r-md);
        padding: 1.25rem;
        margin-bottom: 1.1rem;
        animation: fadeUp 0.6s ease;
    }
    .sidebar-glass h4 {
        font-weight: 700; margin-bottom: 0.9rem;
        display: flex; align-items: center; gap: 8px; font-size: 13.5px;
        letter-spacing: 0.03em; text-transform: uppercase; color: #AFC2F2 !important;
    }
    .sidebar-glass .nav-item {
        padding: 10px 12px; border-radius: 12px; margin: 4px 0;
        display: flex; align-items: center; gap: 10px;
        font-weight: 500; font-size: 14px;
        background: transparent; border-left: 3px solid transparent;
        transition: all 0.2s ease;
    }
    .sidebar-glass .nav-item.active {
        background: rgba(255,255,255,0.12);
        border-left: 3px solid var(--teal);
        font-weight: 700;
    }
    .sidebar-glass .nav-item:hover { background: rgba(255,255,255,0.08); }
    .sidebar-glass p, .sidebar-glass li { font-size: 13.5px; line-height: 1.7; color: #D3DEF7 !important; }
    .sidebar-glass ul { padding-left: 1.1rem; margin: 0; }
    .sidebar-glass .qr-container { display: flex; justify-content: center; margin: 0.7rem 0; }
    .sidebar-glass .qr-container img {
        border-radius: 16px; background: white; padding: 8px;
        box-shadow: var(--shadow-sm);
    }
    .doctor-btn {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.22);
        color: #fff !important; border-radius: 30px; padding: 11px 22px;
        font-weight: 600; font-size: 13.5px; cursor: not-allowed;
        width: 100%; text-align: center; margin-top: 8px;
    }
    .coming-badge {
        background: var(--teal); color: #06312F !important; font-size: 10px; font-weight: 800;
        padding: 3px 10px; border-radius: 30px; margin-left: 6px; letter-spacing: 0.4px;
    }

    /* ===== HERO ===== */
    .hero {
        position: relative;
        background: linear-gradient(120deg, #0B2E86 0%, #1450D6 55%, #1B63E8 100%);
        border-radius: var(--r-xl);
        padding: 2.6rem 2.4rem;
        margin-bottom: 1.6rem;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
        animation: fadeUp 0.7s ease;
    }
    .hero::before, .hero::after {
        content: ""; position: absolute; border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }
    .hero::before { width: 340px; height: 340px; top: -140px; right: -80px; }
    .hero::after { width: 220px; height: 220px; bottom: -100px; left: 10%; background: rgba(14,165,164,0.18); }
    .hero-content {
        position: relative; z-index: 2;
        display: flex; align-items: center; justify-content: space-between;
        gap: 32px; flex-wrap: wrap;
    }
    .hero-visual { position: relative; flex-shrink: 0; width: 190px; height: 190px; margin: 0 auto; }
    .hero-visual .ring {
        position: absolute; inset: 0; border-radius: 50%;
        border: 2.5px solid rgba(255,255,255,0.55);
        animation: pulseRing 2.4s ease-out infinite;
    }
    .doctor-image {
        width: 100%; height: 100%; border-radius: 50%; object-fit: cover;
        border: 4px solid rgba(255,255,255,0.9);
        box-shadow: 0 18px 40px rgba(0,0,0,0.25);
        animation: floatY 4s ease-in-out infinite;
        position: relative; z-index: 2;
    }
    .hero-visual .sparkle {
        position: absolute; bottom: 4px; right: -6px; z-index: 3;
        width: 44px; height: 44px; border-radius: 50%;
        background: rgba(255,255,255,0.22);
        backdrop-filter: blur(6px);
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; box-shadow: var(--shadow-sm);
    }
    .hero-pulse { flex: 1; min-width: 180px; opacity: 0.85; }
    .hero-pulse svg path {
        stroke-dasharray: 12 8;
        animation: dashMove 6s linear infinite;
    }
    .hero-text { flex: 1.3; min-width: 260px; text-align: center; }
    .hero-text h1 {
        font-size: 34px; font-weight: 800; color: #fff; margin: 0 0 6px; line-height: 1.25;
    }
    .hero-text h1 span {
        background: linear-gradient(120deg, #6EF3D6, #7FD8FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-text p { font-size: 15.5px; color: rgba(255,255,255,0.82); margin: 0 0 16px; max-width: 460px; margin-left:auto; margin-right:auto; }
    .hero-badge {
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.28);
        color: #fff; padding: 7px 18px; border-radius: 30px;
        font-size: 13px; font-weight: 600; display: inline-block;
    }

    /* ===== FEATURES ===== */
    .features-grid {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 1.4rem 0;
    }
    @media (max-width: 768px) { .features-grid { grid-template-columns: repeat(2, 1fr); } }
    .feature-card {
        background: var(--surface);
        border-radius: var(--r-md);
        padding: 1.4rem 1.2rem;
        text-align: center;
        border: 1px solid var(--line);
        transition: all 0.25s ease;
        box-shadow: var(--shadow-sm);
    }
    .feature-card:hover { transform: translateY(-5px); box-shadow: var(--shadow-md); border-color: rgba(37,99,235,0.18); }
    .feature-card .icon-circle {
        width: 48px; height: 48px; border-radius: 14px;
        background: var(--primary-100); color: var(--primary-700);
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; margin: 0 auto 10px;
    }
    .feature-card h4 { color: var(--ink); font-weight: 700; margin: 4px 0 4px; font-size: 15.5px; }
    .feature-card p { color: var(--slate); font-size: 13px; margin: 0; line-height: 1.5; }

    /* ===== HOW IT WORKS ===== */
    .how-section {
        background: var(--surface);
        border-radius: var(--r-lg);
        padding: 2.2rem 2rem;
        margin: 1.6rem 0;
        border: 1px solid var(--line);
        box-shadow: var(--shadow-sm);
    }
    .how-title-row { display:flex; align-items:center; justify-content:center; gap:14px; margin-bottom: 1.8rem; }
    .how-title-row .line { height:1px; width:48px; background: var(--line); }
    .how-section h3 { color: var(--ink); font-weight: 800; font-size: 22px; margin: 0; }
    .how-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; position: relative; }
    @media (max-width: 768px) { .how-steps { grid-template-columns: 1fr; } }
    .how-step {
        text-align: center; padding: 1.4rem 1rem;
        background: var(--bg); border-radius: var(--r-md);
        border: 1px solid var(--line);
        transition: all 0.25s ease;
    }
    .how-step:hover { box-shadow: var(--shadow-sm); transform: translateY(-3px); }
    .how-step .step-num {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-900));
        color: white; width: 42px; height: 42px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 18px; margin: 0 auto 12px;
        box-shadow: var(--shadow-sm);
    }
    .how-step h5 { color: var(--ink); font-weight: 700; margin: 0 0 4px; font-size: 15.5px; }
    .how-step p { color: var(--slate); font-size: 13.5px; margin: 0; }

    /* ===== TRUST BADGES ===== */
    .trust-section {
        display: flex; justify-content: center; gap: 34px; flex-wrap: wrap;
        margin: 1.5rem 0; padding: 1.3rem 1.5rem;
        background: var(--primary-100);
        border: 1px solid rgba(37,99,235,0.1);
        border-radius: var(--r-md);
    }
    .trust-item { display: flex; align-items: center; gap: 8px; font-weight: 600; color: var(--primary-900); font-size: 14px; }
    .trust-item .icon { font-size: 20px; }

    /* ===== UPLOAD ===== */
    .upload-card {
        background: var(--surface);
        border-radius: var(--r-xl);
        padding: 2.2rem 1.5rem;
        text-align: center;
        border: 2px dashed rgba(37,99,235,0.28);
        transition: all 0.3s ease;
        margin-bottom: 1.4rem;
        box-shadow: var(--shadow-sm);
    }
    .upload-card:hover { border-color: var(--primary-600); box-shadow: var(--shadow-md); }
    .upload-card .icon-circle {
        width: 64px; height: 64px; border-radius: 18px;
        background: var(--primary-100); color: var(--primary-700);
        display: flex; align-items: center; justify-content: center;
        font-size: 28px; margin: 0 auto 12px;
        animation: floatY 3.5s ease-in-out infinite;
    }
    .upload-card h3 { font-weight: 800; color: var(--ink); margin: 4px 0 4px; font-size: 19px; }
    .upload-card p { color: var(--slate); font-size: 13.5px; margin: 0; }

    /* ===== PREVIEW ===== */
    .preview-container {
        background: var(--surface);
        border-radius: var(--r-lg);
        padding: 1.4rem;
        border: 1px solid var(--line);
        box-shadow: var(--shadow-sm);
        margin: 1.4rem 0;
        animation: fadeUp 0.5s ease;
    }
    .preview-container h4 {
        display:flex; align-items:center; gap:8px; color: var(--ink);
        font-weight: 700; font-size: 15px; margin-bottom: 0.6rem;
    }
    .preview-container img { border-radius: 14px; max-height: 300px; object-fit: contain; width: 100%; }

    /* ===== SECTION TITLE ===== */
    .section-title {
        font-size: 20px; font-weight: 800; color: var(--ink);
        margin: 1.8rem 0 1rem; padding: 0 0 0 12px;
        border-left: 4px solid var(--primary-600);
        display: flex; align-items: center; gap: 8px;
        animation: fadeUp 0.4s ease;
    }

    /* ===== RESULT CARDS ===== */
    .result-card {
        border-radius: var(--r-lg);
        padding: 1.7rem 1.8rem;
        margin-top: 0.4rem;
        border: 1px solid transparent;
        display: flex; align-items: center; gap: 16px;
        animation: fadeUp 0.4s ease;
        box-shadow: var(--shadow-sm);
    }
    .result-card .badge-ic {
        width: 54px; height: 54px; border-radius: 16px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center; font-size: 26px;
    }
    .result-card.positive { background: var(--danger-100); border-color: rgba(226,63,78,0.16); }
    .result-card.positive .badge-ic { background: var(--danger); color: #fff; }
    .result-card.negative { background: var(--success-100); border-color: rgba(16,168,90,0.16); }
    .result-card.negative .badge-ic { background: var(--success); color: #fff; }
    .result-card h2 { font-size: 22px; font-weight: 800; margin: 0 0 4px; color: var(--ink); }
    .result-card .confidence { font-size: 15px; font-weight: 600; color: var(--ink); }
    .result-card .confidence strong { color: var(--primary-700); }
    .result-card .sub { font-size: 13px; color: var(--slate); margin-top: 2px; }

    /* ===== DISCLAIMER ===== */
    .disclaimer {
        background: var(--amber-100);
        border-radius: var(--r-md);
        padding: 1.1rem 1.5rem;
        border: 1px solid rgba(222,154,31,0.22);
        border-left: 5px solid var(--amber);
        margin-top: 1.8rem;
        font-size: 13px; color: #6B4E12; line-height: 1.7;
        animation: fadeUp 0.5s ease;
    }
    .disclaimer strong { color: #573B08; }

    /* ===== GENERAL / WIDGETS ===== */
    .stImage img { border-radius: 14px; box-shadow: var(--shadow-sm); }

    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: var(--ink) !important; }
    [data-testid="stMetric"] {
        background: var(--surface);
        border-radius: 14px; padding: 12px 16px;
        border: 1px solid var(--line);
        box-shadow: var(--shadow-sm);
    }

    #MainMenu, footer, .stDeployButton { display: none; }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-900));
        color: white; border: none; border-radius: 40px;
        padding: 13px 28px; font-weight: 700; letter-spacing: 0.01em;
        transition: 0.25s; width: 100%;
        box-shadow: 0 10px 24px rgba(20,80,214,0.24);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(20,80,214,0.32); }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-600), var(--teal));
        border-radius: 10px; transition: width 0.5s ease;
    }

    .stDataFrame, [data-testid="stDataFrame"] {
        border-radius: 14px; overflow: hidden; border: 1px solid var(--line);
    }

    [data-testid="stExpander"] {
        background: var(--surface); border-radius: 14px; border: 1px solid var(--line);
        box-shadow: var(--shadow-sm);
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE ==========
if 'language' not in st.session_state:
    st.session_state.language = "fr"
if 'history' not in st.session_state:
    st.session_state.history = []

# ========== HEADER ==========
def get_logo_base64():
    for name in ["logo.png", "logo.jpg", "logo.jpeg", "LOGO.png"]:
        if os.path.exists(name):
            with open(name, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None

logo = get_logo_base64()

# ========== SIDEBAR ==========
with st.sidebar:
    # Sélecteur de langue
    lang_map = {
        "ar": "🇸🇦 العربية",
        "fr": "🇫🇷 Français",
        "en": "🇬🇧 English"
    }
    selected_lang = st.selectbox(
        t("lang_selector"),
        options=list(lang_map.keys()),
        format_func=lambda x: lang_map[x],
        index=list(lang_map.keys()).index(st.session_state.language)
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()
    
    st.markdown("---")
    
    # ===== قائمة ثابتة =====
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>📋 القائمة / Menu</h4>
        <div class="nav-item active">🏠 {t('nav_home')}</div>
        <div class="nav-item">🔬 {t('nav_analyze')}</div>
        <div class="nav-item">📊 {t('nav_history')}</div>
        <div class="nav-item">ℹ️ {t('nav_info')}</div>
        <div class="nav-item">💡 {t('nav_health')}</div>
        <div class="nav-item">📱 {t('nav_about')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # QR Code
    APP_URL = "https://hwaxrexkahkxaazwwjjr3d.streamlit.app/"
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&margin=10&color=1450D6&data={APP_URL}"
    
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>{t('sidebar_qr_title')}</h4>
        <div class="qr-container">
            <img src="{qr_api_url}" width="160" height="160" alt="QR Code">
        </div>
        <p style="text-align:center; font-size:13px; color:#AFC2F2;">{t('sidebar_qr_desc')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Conseils santé
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>{t('sidebar_health_title')}</h4>
        <ul>
            <li>{t('sidebar_health_1')}</li>
            <li>{t('sidebar_health_2')}</li>
            <li>{t('sidebar_health_3')}</li>
            <li>{t('sidebar_health_4')}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Consultation médecin
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>{t('sidebar_doctor_title')} <span class="coming-badge">{t('sidebar_doctor_soon')}</span></h4>
        <p>{t('sidebar_doctor_desc')}</p>
        <div class="doctor-btn">{t('sidebar_doctor_btn')}</div>
        <p style="font-size:12px; color:#7C93C9; text-align:center; margin-top:8px;">{t('sidebar_doctor_soon')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption(t('sidebar_version'))

# ========== PAGE PRINCIPALE ==========
# Navbar avec badges
logo_mark_html = f'<img src="data:image/png;base64,{logo}" class="mark-img">' if logo else '<div class="mark">🩸</div>'

st.markdown(f"""
<div class="navbar">
    <div class="navbar-brand">
        {logo_mark_html}
        <div>
            <h1>{t('app_title')}</h1>
            <div class="subtitle">{t('app_subtitle')}</div>
        </div>
    </div>
    <div class="navbar-badges">
        <div class="navbar-chip">
            <div class="chip-ic">🔒</div>
            <div class="chip-text">
                <span class="chip-title">{t('badge_private')}</span>
                <span class="chip-sub">{t('badge_private_desc')}</span>
            </div>
        </div>
        <div class="navbar-chip">
            <div class="chip-ic">⏰</div>
            <div class="chip-text">
                <span class="chip-title">{t('badge_available')}</span>
                <span class="chip-sub">{t('badge_available_desc')}</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== HERO SECTION AVEC IMAGE DU MÉDECIN ==========
# Utilisation d'une icône de médecin (Flaticon)
doctor_img_url = "https://cdn-icons-png.flaticon.com/512/3774/3774299.png"

st.markdown(f"""
<div class="hero">
    <div class="hero-content">
        <div class="hero-visual">
            <div class="ring"></div>
            <img src="{doctor_img_url}" class="doctor-image" alt="Médecin / Doctor">
            <div class="sparkle">✨</div>
        </div>
        <div class="hero-pulse">
            <svg viewBox="0 0 220 60" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none" style="width:100%;height:44px;">
                <path d="M0 30 L45 30 L58 8 L72 52 L86 30 L220 30" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div class="hero-text">
            <h1>{t('hero_title')}</h1>
            <p>{t('hero_desc')}</p>
            <span class="hero-badge">✅ {t('hero_badge')}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== ZONE UPLOAD ==========
st.markdown(f"""
<div class="upload-card">
    <div class="icon-circle">📸</div>
    <h3>{t('upload_title')}</h3>
    <p>{t('upload_desc')}</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])
with col1:
    option = st.radio(
        t("upload_method_file"),
        [t("upload_method_file"), t("upload_method_camera")],
        horizontal=True,
        label_visibility="collapsed"
    )
with col2:
    if option == t("upload_method_file"):
        uploaded = st.file_uploader(
            t("upload_file_label"),
            type=["jpg","png","jpeg"],
            label_visibility="collapsed",
            key="file_uploader"
        )
    else:
        uploaded = st.camera_input(
            t("upload_camera_label"),
            label_visibility="collapsed",
            key="camera_input"
        )

# ========== FONCTIONS ==========
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

@st.cache_resource
def load_models():
    unet, dev_unet = load_unet_model()
    clf, dev_clf = load_classifier_model()
    return unet, dev_unet, clf, dev_clf

# ========== TRAITEMENT ==========
if uploaded is not None:
    st.markdown(f"""
    <div class="preview-container">
        <h4 style="margin-top:0;">{t('preview_title')}</h4>
    """, unsafe_allow_html=True)
    col_preview, _ = st.columns([1, 1])
    with col_preview:
        st.image(uploaded, use_container_width=True, caption=t('preview_caption'))
    st.markdown("</div>", unsafe_allow_html=True)
    
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

            conj_enhanced, final_mask, bbox = extract_best_conjunctiva(img, raw_mask)

            result, confidence, raw_pred = predict_anemia(clf_model, conj_enhanced, clf_device)

            anemia_pct = raw_pred * 100
            non_pct = (1 - raw_pred) * 100

            progress_bar.empty()
            st.success(t("analysis_done"))

            # --- IMAGES ---
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

            # --- MÉTRIQUES ---
            before = np.sum(raw_mask > 0) / 255
            after = np.sum(final_mask > 0) / 255
            reduction = ((before - after) / before * 100) if before > 0 else 0

            m1, m2 = st.columns(2)
            with m1:
                st.metric(t("metric_surface"), f"{after:.0f} px²")
            with m2:
                st.metric(t("metric_cleaning"), f"{reduction:.1f}%")

            # --- DIAGNOSTIC ---
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

            # --- GRAPHIQUE ---
            st.markdown(f'<div class="section-title">{t("chart_title")}</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8,5))
            cats = [t('chart_non'), t('chart_anemic')]
            vals = [non_pct, anemia_pct]
            colors = ['#10b981', '#dc2626']
            bars = ax.bar(cats, vals, color=colors, width=0.5, edgecolor='white', linewidth=2)
            ax.set_ylim(0,100)
            ax.set_ylabel('Pourcentage (%)')
            ax.set_title(t('chart_title'), fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)

            # --- HISTORIQUE ---
            entry = {
                t("history_date"): datetime.now().strftime("%Y-%m-%d %H:%M"),
                t("history_diagnostic"): result,
                t("history_confidence"): f"{confidence:.1f}%",
                t("history_prob"): f"{anemia_pct:.1f}%"
            }
            st.session_state.history.append(entry)
            if len(st.session_state.history) > 10:
                st.session_state.history.pop(0)

            if st.session_state.history:
                st.markdown(f'<div class="section-title">{t("history_title")}</div>', unsafe_allow_html=True)
                df = pd.DataFrame(st.session_state.history)
                st.dataframe(df, use_container_width=True, hide_index=True)

            # --- DÉTAILS TECHNIQUES ---
            with st.expander(t("tech_details")):
                st.write(f"**{t('tech_model_seg')}:** U‑Net (ResNet34)")
                st.write(f"**{t('tech_model_clf')}:** EfficientNet‑B3")
                st.write(f"**{t('tech_device')}:** {'GPU' if clf_device.type == 'cuda' else 'CPU'}")
                st.write(f"**{t('tech_sigmoid')}:** {raw_pred:.4f}")
                st.write(f"**{t('tech_prob_anemic')}:** {anemia_pct:.1f}%")
                st.write(f"**{t('tech_prob_non')}:** {non_pct:.1f}%")
                st.write(f"**{t('tech_preprocess')}:** CLAHE + Filtrage + Netteté")
                st.write(f"**{t('tech_decision')}:** {t('tech_decision_value')}")

            # --- AVERTISSEMENT ---
            st.markdown(f"""
            <div class="disclaimer">
                ⚠️ <strong>{t('disclaimer')}</strong><br>
                {t('disclaimer_text')}<br>
                {t('disclaimer_consult')}
            </div>
            """, unsafe_allow_html=True)

# ========== FEATURES GRID ==========
st.markdown("""
<div class="features-grid">
    <div class="feature-card">
        <div class="icon-circle">⚡</div>
        <h4>""" + t('feature_1_title') + """</h4>
        <p>""" + t('feature_1_desc') + """</p>
    </div>
    <div class="feature-card">
        <div class="icon-circle">📤</div>
        <h4>""" + t('feature_2_title') + """</h4>
        <p>""" + t('feature_2_desc') + """</p>
    </div>
    <div class="feature-card">
        <div class="icon-circle">✅</div>
        <h4>""" + t('feature_3_title') + """</h4>
        <p>""" + t('feature_3_desc') + """</p>
    </div>
    <div class="feature-card">
        <div class="icon-circle">🔒</div>
        <h4>""" + t('feature_4_title') + """</h4>
        <p>""" + t('feature_4_desc') + """</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== HOW IT WORKS ==========
st.markdown(f"""
<div class="how-section">
    <div class="how-title-row">
        <div class="line"></div>
        <h3>{t('how_title')}</h3>
        <div class="line"></div>
    </div>
    <div class="how-steps">
        <div class="how-step">
            <div class="step-num">1</div>
            <h5>{t('how_step1')}</h5>
            <p>{t('how_step1_desc')}</p>
        </div>
        <div class="how-step">
            <div class="step-num">2</div>
            <h5>{t('how_step2')}</h5>
            <p>{t('how_step2_desc')}</p>
        </div>
        <div class="how-step">
            <div class="step-num">3</div>
            <h5>{t('how_step3')}</h5>
            <p>{t('how_step3_desc')}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== TRUST BADGES ==========
st.markdown(f"""
<div class="trust-section">
    <div class="trust-item"><span class="icon">🔒</span> {t('trust_1')}</div>
    <div class="trust-item"><span class="icon">❤️</span> {t('trust_2')}</div>
    <div class="trust-item"><span class="icon">⭐</span> {t('trust_3')}</div>
    <div class="trust-item"><span class="icon">📱</span> {t('version')}</div>
    <div class="trust-item"><span class="icon">🎯</span> {t('accuracy')}</div>
</div>
""", unsafe_allow_html=True)

# ========== BOTTOM INFO ==========
st.markdown(f"""
<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:1.5rem 0;">
    <div class="feature-card">
        <div class="icon-circle">⚡</div>
        <h4>{t('fast')}</h4>
        <p>{t('fast_desc')}</p>
    </div>
    <div class="feature-card">
        <div class="icon-circle">🧠</div>
        <h4>{t('smart')}</h4>
        <p>{t('smart_desc')}</p>
    </div>
    <div class="feature-card">
        <div class="icon-circle">🔒</div>
        <h4>{t('secure')}</h4>
        <p>{t('secure_desc')}</p>
    </div>
</div>
""", unsafe_allow_html=True)
