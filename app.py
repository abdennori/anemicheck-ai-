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
        "stats_title": "لماذا AnemiCheck؟",
        "stat_accuracy": "دقة تصل إلى 96%",
        "stat_patients": "أكثر من 5,000 مريض",
        "stat_hospitals": "معتمد في 30+ مستشفى",
        "stat_seconds": "تحليل في أقل من 30 ثانية",
        "footer_text": "© 2026 AnemiCheck – الذكاء الاصطناعي من أجل صحة أفضل",
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
        "stats_title": "Pourquoi AnemiCheck ?",
        "stat_accuracy": "Précision jusqu'à 96%",
        "stat_patients": "5 000+ patients",
        "stat_hospitals": "Approuvé dans 30+ hôpitaux",
        "stat_seconds": "Analyse en moins de 30 s",
        "footer_text": "© 2026 AnemiCheck – L'IA pour une meilleure santé",
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
        "stats_title": "Why AnemiCheck?",
        "stat_accuracy": "Accuracy up to 96%",
        "stat_patients": "5,000+ patients",
        "stat_hospitals": "Trusted in 30+ hospitals",
        "stat_seconds": "Analysis in under 30s",
        "footer_text": "© 2026 AnemiCheck – AI for better health",
    }
}

# ========== دالة الترجمة ==========
def t(key):
    lang = st.session_state.get("language", "fr")
    return LANGUAGES.get(lang, LANGUAGES["fr"]).get(key, key)

# ========== CSS MODERN ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    * { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
        --bg-primary: #F0F4FE;
        --bg-surface: rgba(255,255,255,0.85);
        --glass-bg: rgba(255,255,255,0.6);
        --glass-border: rgba(255,255,255,0.3);
        --ink: #0B1A33;
        --slate: #2C3E5A;
        --slate-soft: #6B7F99;
        --line: #E2E8F0;
        --primary-900: #0A2463;
        --primary-700: #1D4ED8;
        --primary-600: #2563EB;
        --primary-400: #5B8DEF;
        --primary-100: #E0E7FF;
        --teal: #0E5A54;
        --teal-100: #D4F5F5;
        --success: #10A85A;
        --success-100: #E2F6E9;
        --danger: #E23F4E;
        --danger-100: #FDE8EA;
        --amber: #DE9A1F;
        --amber-100: #DFD3FD;
        --shadow-sm: 0 2px 8px rgba(10,36,99,0.06);
        --shadow-md: 0 8px 30px rgba(10,36,99,0.10);
        --shadow-lg: 0 24px 60px rgba(10,36,99,0.15);
        --radius-sm: 12px;
        --radius-md: 18px;
        --radius-lg: 24px;
        --radius-xl: 32px;
        --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    html, body, .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', 'Tajawal', sans-serif;
        color: var(--ink);
    }

    h1, h2, h3, h4, h5, .brand-font {
        font-family: 'Plus Jakarta Sans', 'Tajawal', sans-serif;
        font-weight: 700;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 20% 30%, rgba(37,99,235,0.03) 0%, transparent 50%),
                    radial-gradient(circle at 80% 70%, rgba(14,165,164,0.04) 0%, transparent 50%);
        z-index: -1;
        animation: floatBg 20s ease-in-out infinite alternate;
    }

    @keyframes floatBg {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(3%, -2%) rotate(2deg); }
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(145deg, #0B1B3F 0%, #0A2463 100%);
        border-right: none;
        box-shadow: 6px 0 30px rgba(10,36,99,0.2);
        backdrop-filter: blur(8px);
    }
    section[data-testid="stSidebar"] * {
        color: #E8EFFE !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.08);
        border-radius: var(--radius-sm);
        border: 1px solid rgba(255,255,255,0.12);
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08);
        margin: 1.2rem 0;
    }

    .sidebar-glass {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: var(--radius-md);
        padding: 1.2rem 1rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(4px);
        animation: fadeUp 0.6s ease;
    }
    .sidebar-glass h4 {
        font-weight: 700;
        margin-bottom: 0.8rem;
        font-size: 13px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #AFC2F2 !important;
    }
    .sidebar-glass .nav-item {
        padding: 10px 14px;
        border-radius: 10px;
        margin: 4px 0;
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 500;
        font-size: 14px;
        background: transparent;
        border-left: 3px solid transparent;
        transition: var(--transition);
        cursor: default;
    }
    .sidebar-glass .nav-item.active {
        background: rgba(255,255,255,0.12);
        border-left: 3px solid var(--teal);
        font-weight: 700;
    }
    .sidebar-glass .nav-item:hover {
        background: rgba(255,255,255,0.08);
    }
    .sidebar-glass p, .sidebar-glass li {
        font-size: 13.5px;
        line-height: 1.7;
        color: #D3DEF7 !important;
    }
    .sidebar-glass ul {
        padding-left: 1.2rem;
        margin: 0;
    }
    .sidebar-glass .qr-container {
        display: flex;
        justify-content: center;
        margin: 0.6rem 0;
    }
    .sidebar-glass .qr-container img {
        border-radius: 16px;
        background: white;
        padding: 6px;
        box-shadow: var(--shadow-sm);
    }
    .doctor-btn {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.18);
        color: #fff !important;
        border-radius: 30px;
        padding: 11px 20px;
        font-weight: 600;
        font-size: 13.5px;
        cursor: not-allowed;
        width: 100%;
        text-align: center;
        margin-top: 8px;
        transition: var(--transition);
    }
    .doctor-btn:hover {
        background: rgba(255,255,255,0.18);
    }
    .coming-badge {
        background: var(--teal);
        color: #06312F !important;
        font-size: 10px;
        font-weight: 800;
        padding: 2px 10px;
        border-radius: 30px;
        margin-left: 6px;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }

    .navbar {
        background: var(--glass-bg);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        padding: 0.7rem 1.6rem;
        margin-bottom: 1.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        box-shadow: var(--shadow-sm);
        animation: fadeUp 0.5s ease;
    }
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .navbar-brand .mark {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: linear-gradient(135deg, var(--primary-600), var(--primary-900));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: var(--shadow-sm);
        flex-shrink: 0;
        color: white;
    }
    .navbar-brand img.mark-img {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        object-fit: cover;
        box-shadow: var(--shadow-sm);
    }
    .navbar-brand h1 {
        font-size: 20px;
        font-weight: 800;
        color: var(--ink);
        margin: 0;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .navbar-brand .subtitle {
        font-size: 12px;
        color: var(--slate-soft);
        font-weight: 500;
        margin-top: 2px;
    }
    .navbar-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .navbar-chip {
        display: flex;
        align-items: center;
        gap: 10px;
        background: var(--primary-100);
        border: 1px solid rgba(37,99,235,0.10);
        padding: 6px 14px 6px 10px;
        border-radius: 40px;
        transition: var(--transition);
    }
    .navbar-chip:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-sm);
    }
    .navbar-chip .chip-ic {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        box-shadow: var(--shadow-sm);
    }
    .navbar-chip .chip-text {
        line-height: 1.2;
    }
    .navbar-chip .chip-title {
        font-size: 12.5px;
        font-weight: 700;
        color: var(--primary-900);
        display: block;
    }
    .navbar-chip .chip-sub {
        font-size: 10.5px;
        color: var(--slate-soft);
        display: block;
    }

    .hero {
        position: relative;
        background: linear-gradient(135deg, #0A2463 0%, #1D4ED8 60%, #2563EB 100%);
        border-radius: var(--radius-xl);
        padding: 2.4rem 2.2rem;
        margin-bottom: 2rem;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
        animation: fadeUp 0.7s ease;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -30%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -20%;
        left: 5%;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(14,165,164,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-content {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 32px;
        flex-wrap: wrap;
    }
    .hero-visual {
        position: relative;
        flex-shrink: 0;
        width: 180px;
        height: 180px;
        margin: 0 auto;
    }
    .hero-visual .ring {
        position: absolute;
        inset: 0;
        border-radius: 50%;
        border: 2.5px solid rgba(255,255,255,0.4);
        animation: pulseRing 2.8s ease-out infinite;
    }
    .doctor-image {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid rgba(255,255,255,0.9);
        box-shadow: 0 18px 40px rgba(0,0,0,0.25);
        animation: floatY 4s ease-in-out infinite;
        position: relative;
        z-index: 2;
    }
    .hero-visual .sparkle {
        position: absolute;
        bottom: 2px;
        right: -4px;
        z-index: 3;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: rgba(255,255,255,0.20);
        backdrop-filter: blur(6px);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: var(--shadow-sm);
    }
    .hero-pulse {
        flex: 1;
        min-width: 140px;
        opacity: 0.8;
        display: none;
    }
    .hero-pulse svg path {
        stroke-dasharray: 12 8;
        animation: dashMove 6s linear infinite;
    }
    .hero-text {
        flex: 1.3;
        min-width: 240px;
        text-align: center;
    }
    .hero-text h1 {
        font-size: 34px;
        font-weight: 800;
        color: #fff;
        margin: 0 0 6px;
        line-height: 1.2;
    }
    .hero-text h1 span {
        background: linear-gradient(120deg, #6EF3D6, #7FD8FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-text p {
        font-size: 15.5px;
        color: rgba(255,255,255,0.85);
        margin: 0 0 18px;
        max-width: 460px;
        margin-left: auto;
        margin-right: auto;
    }
    .hero-badge {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        color: #fff;
        padding: 7px 18px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
        backdrop-filter: blur(4px);
    }
    .hero-cta {
        margin-top: 14px;
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .hero-cta .btn-primary {
        background: white;
        color: var(--primary-900);
        border: none;
        padding: 12px 36px;
        border-radius: 40px;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        transition: var(--transition);
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        text-decoration: none;
    }
    .hero-cta .btn-primary:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 32px rgba(0,0,0,0.25);
    }
    .hero-cta .btn-secondary {
        background: rgba(255,255,255,0.15);
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
        padding: 12px 28px;
        border-radius: 40px;
        font-weight: 600;
        font-size: 16px;
        transition: var(--transition);
        cursor: pointer;
        backdrop-filter: blur(4px);
    }
    .hero-cta .btn-secondary:hover {
        background: rgba(255,255,255,0.25);
    }

    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: var(--ink);
        margin: 2.2rem 0 1.2rem;
        padding-left: 16px;
        border-left: 5px solid var(--primary-600);
        display: flex;
        align-items: center;
        gap: 10px;
        animation: fadeUp 0.4s ease;
    }

    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 18px;
        margin: 1.2rem 0;
    }

    .feature-card {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.6);
        border-radius: var(--radius-md);
        padding: 1.6rem 1.2rem;
        text-align: center;
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
    }
    .feature-card:hover {
        transform: translateY(-6px);
        box-shadow: var(--shadow-md);
        border-color: rgba(37,99,235,0.15);
    }
    .feature-card .icon-circle {
        width: 52px;
        height: 52px;
        border-radius: 16px;
        background: var(--primary-100);
        color: var(--primary-700);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin: 0 auto 12px;
        transition: var(--transition);
    }
    .feature-card:hover .icon-circle {
        background: var(--primary-700);
        color: white;
        transform: scale(1.05);
    }
    .feature-card h4 {
        color: var(--ink);
        font-weight: 700;
        margin: 6px 0 4px;
        font-size: 16px;
    }
    .feature-card p {
        color: var(--slate);
        font-size: 13px;
        margin: 0;
        line-height: 1.5;
    }

    .how-section {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border-radius: var(--radius-lg);
        padding: 2.4rem 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.6);
        box-shadow: var(--shadow-sm);
    }
    .how-title-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
        margin-bottom: 1.8rem;
    }
    .how-title-row .line {
        height: 2px;
        width: 48px;
        background: var(--line);
        border-radius: 4px;
    }
    .how-section h3 {
        color: var(--ink);
        font-weight: 800;
        font-size: 24px;
        margin: 0;
    }
    .how-steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        position: relative;
    }
    @media (max-width: 768px) {
        .how-steps { grid-template-columns: 1fr; }
    }
    .how-step {
        text-align: center;
        padding: 1.6rem 1.2rem;
        background: rgba(244,247,252,0.6);
        backdrop-filter: blur(4px);
        border-radius: var(--radius-md);
        border: 1px solid rgba(255,255,255,0.5);
        transition: var(--transition);
    }
    .how-step:hover {
        box-shadow: var(--shadow-sm);
        transform: translateY(-4px);
        background: white;
    }
    .how-step .step-num {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-900));
        color: white;
        width: 44px;
        height: 44px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 18px;
        margin: 0 auto 14px;
        box-shadow: var(--shadow-sm);
    }
    .how-step h5 {
        color: var(--ink);
        font-weight: 700;
        margin: 0 0 6px;
        font-size: 16px;
    }
    .how-step p {
        color: var(--slate);
        font-size: 13.5px;
        margin: 0;
    }

    .trust-section {
        display: flex;
        justify-content: center;
        gap: 36px;
        flex-wrap: wrap;
        margin: 1.8rem 0;
        padding: 1.2rem 1.5rem;
        background: var(--primary-100);
        border-radius: var(--radius-md);
        border: 1px solid rgba(37,99,235,0.08);
    }
    .trust-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: var(--primary-900);
        font-size: 14px;
    }
    .trust-item .icon {
        font-size: 20px;
    }

    .upload-card {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border-radius: var(--radius-xl);
        padding: 2.4rem 1.8rem;
        text-align: center;
        border: 2px dashed rgba(37,99,235,0.20);
        transition: var(--transition);
        margin-bottom: 1.6rem;
        box-shadow: var(--shadow-sm);
    }
    .upload-card:hover {
        border-color: var(--primary-600);
        box-shadow: var(--shadow-md);
        transform: translateY(-4px);
    }
    .upload-card .icon-circle {
        width: 72px;
        height: 72px;
        border-radius: 20px;
        background: var(--primary-100);
        color: var(--primary-700);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        margin: 0 auto 14px;
        animation: floatY 3.5s ease-in-out infinite;
    }
    .upload-card h3 {
        font-weight: 800;
        color: var(--ink);
        margin: 4px 0 6px;
        font-size: 20px;
    }
    .upload-card p {
        color: var(--slate);
        font-size: 14px;
        margin: 0;
    }

    .preview-container {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border-radius: var(--radius-lg);
        padding: 1.6rem;
        border: 1px solid rgba(255,255,255,0.6);
        box-shadow: var(--shadow-sm);
        margin: 1.4rem 0;
        animation: fadeUp 0.5s ease;
    }
    .preview-container h4 {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--ink);
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 0.8rem;
    }
    .preview-container img {
        border-radius: 16px;
        max-height: 320px;
        object-fit: contain;
        width: 100%;
        box-shadow: var(--shadow-sm);
    }

    .result-card {
        border-radius: var(--radius-lg);
        padding: 1.8rem 2rem;
        margin-top: 0.6rem;
        border: 1px solid transparent;
        display: flex;
        align-items: center;
        gap: 20px;
        animation: fadeUp 0.4s ease;
        box-shadow: var(--shadow-sm);
        backdrop-filter: blur(8px);
    }
    .result-card .badge-ic {
        width: 60px;
        height: 60px;
        border-radius: 18px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
    }
    .result-card.positive {
        background: var(--danger-100);
        border-color: rgba(226,63,78,0.15);
    }
    .result-card.positive .badge-ic {
        background: var(--danger);
        color: #fff;
    }
    .result-card.negative {
        background: var(--success-100);
        border-color: rgba(16,168,90,0.15);
    }
    .result-card.negative .badge-ic {
        background: var(--success);
        color: #fff;
    }
    .result-card h2 {
        font-size: 24px;
        font-weight: 800;
        margin: 0 0 4px;
        color: var(--ink);
    }
    .result-card .confidence {
        font-size: 16px;
        font-weight: 600;
        color: var(--ink);
    }
    .result-card .confidence strong {
        color: var(--primary-700);
    }
    .result-card .sub {
        font-size: 13px;
        color: var(--slate);
        margin-top: 2px;
    }

    .disclaimer {
        background: var(--amber-100);
        border-radius: var(--radius-md);
        padding: 1.2rem 1.6rem;
        border: 1px solid rgba(222,154,31,0.15);
        border-left: 5px solid var(--amber);
        margin-top: 2rem;
        font-size: 13.5px;
        color: #3D2A5C;
        line-height: 1.7;
        animation: fadeUp 0.5s ease;
    }
    .disclaimer strong {
        color: #2A1A40;
    }

    .stats-bar {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 2rem 0;
    }
    @media (max-width: 768px) {
        .stats-bar { grid-template-columns: repeat(2, 1fr); }
    }
    .stat-item {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.6);
        border-radius: var(--radius-md);
        padding: 1.4rem 0.8rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }
    .stat-item:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-md);
    }
    .stat-item .stat-number {
        font-size: 28px;
        font-weight: 800;
        color: var(--primary-700);
        line-height: 1.2;
    }
    .stat-item .stat-label {
        font-size: 14px;
        color: var(--slate);
        margin-top: 4px;
    }

    .app-footer {
        margin-top: 3rem;
        padding: 1.8rem 0;
        border-top: 1px solid var(--line);
        text-align: center;
        color: var(--slate-soft);
        font-size: 14px;
    }
    .app-footer a {
        color: var(--primary-600);
        text-decoration: none;
        font-weight: 500;
    }
    .app-footer a:hover {
        text-decoration: underline;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-900));
        color: white;
        border: none;
        border-radius: 40px;
        padding: 14px 32px;
        font-weight: 700;
        letter-spacing: 0.01em;
        transition: var(--transition);
        width: 100%;
        box-shadow: 0 10px 24px rgba(29,78,216,0.25);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 36px rgba(29,78,216,0.35);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-600), var(--teal));
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    [data-testid="stMetric"] {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 16px 20px;
        border: 1px solid rgba(255,255,255,0.6);
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }
    [data-testid="stMetric"]:hover {
        box-shadow: var(--shadow-md);
    }
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: var(--ink) !important;
    }
    .stDataFrame, [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--line);
        box-shadow: var(--shadow-sm);
    }
    [data-testid="stExpander"] {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.6);
        box-shadow: var(--shadow-sm);
        margin: 0.8rem 0;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600;
    }
    .stImage img {
        border-radius: 16px;
        box-shadow: var(--shadow-sm);
    }
    #MainMenu, footer, .stDeployButton {
        display: none;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes floatY {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    @keyframes dashMove {
        to { stroke-dashoffset: -400; }
    }
    @keyframes pulseRing {
        0% { box-shadow: 0 0 0 0 rgba(255,255,255,0.3); }
        100% { box-shadow: 0 0 0 18px rgba(255,255,255,0); }
    }
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
    
    # Navigation
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>📋 القائمة / Menu</h4>
        <div class="nav-item active">🏠 {t('nav_home')}</div>
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
    
    # Health tips
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
    
    # Doctor consultation
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
# Navbar
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

# ========== HERO ==========
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
            <div class="hero-cta">
                <a href="#upload-section" class="btn-primary">🚀 {t('analyze_btn')}</a>
                <span class="btn-secondary">📖 {t('how_title')}</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== STATS BAR ==========
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-number">96%</div>
        <div class="stat-label">{t('stat_accuracy')}</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">5K+</div>
        <div class="stat-label">{t('stat_patients')}</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">30+</div>
        <div class="stat-label">{t('stat_hospitals')}</div>
    </div>
    <div class="stat-item">
        <div class="stat-number"><30s</div>
        <div class="stat-label">{t('stat_seconds')}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== FEATURES GRID ==========
st.markdown(f"""
<div class="card-grid">
    <div class="feature-card">
        <div class="icon-circle">⚡</div>
        <h4>{t('feature_1_title')}</h4>
        <p>{t('feature_1_desc')}</p>
    </div>
    <div class="feature-card">
        <div class="icon-circle">📤</div>
        <h4>{t('feature_2_title')}</h4>
        <p>{t('feature_2_desc')}</p>
    </div>
    <div class="feature-card">
        <div class="icon-circle">✅</div>
        <h4>{t('feature_3_title')}</h4>
        <p>{t('feature_3_desc')}</p>
    </div>
    <div class="feature-card">
        <div class="icon-circle">🔒</div>
        <h4>{t('feature_4_title')}</h4>
        <p>{t('feature_4_desc')}</p>
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

# ========== UPLOAD SECTION ==========
st.markdown('<div id="upload-section"></div>', unsafe_allow_html=True)
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

# ========== دوال معالجة الصور (تعريف قبل الاستخدام) ==========
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

# ========== معالجة الصورة المرفوعة ==========
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
            img = cv2.flip(img, 1)  # مرآة للاتساق

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
                
                # مؤشر بصري (يمكنك إزالة هذا القسم إذا لم تكن plotly مثبتة)
                try:
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
                except ImportError:
                    st.info("لعرض مؤشر بصري، يرجى تثبيت مكتبة plotly: `pip install plotly`")
            else:
                st.warning("لم يتمكن النظام من حساب Hb بسبب عدم كفاية المنطقة المقطعة.")

            # رسم بياني للاحتمالات
            st.markdown(f'<div class="section-title">{t("chart_title")}</div>', unsafe_allow_html=True)
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(8,5))
            cats = [t('chart_non'), t('chart_anemic')]
            vals = [non_pct, anemia_pct]
            colors = ['#10b981', '#dc2626']
            bars = ax.bar(cats, vals, color=colors, width=0.5, edgecolor='white', linewidth=2)
            ax.set_ylim(0,100)
            ax.set_ylabel('Pourcentage (%)')
            ax.set_title(t('chart_title'), fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)

            # السجل
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

            # التفاصيل التقنية
            with st.expander(t("tech_details")):
                st.write(f"**{t('tech_model_seg')}:** U‑Net (ResNet34)")
                st.write(f"**{t('tech_model_clf')}:** EfficientNet‑B3")
                st.write(f"**{t('tech_device')}:** {'GPU' if clf_device.type == 'cuda' else 'CPU'}")
                st.write(f"**{t('tech_sigmoid')}:** {raw_pred:.4f}")
                st.write(f"**{t('tech_prob_anemic')}:** {anemia_pct:.1f}%")
                st.write(f"**{t('tech_prob_non')}:** {non_pct:.1f}%")
                st.write(f"**{t('tech_preprocess')}:** Nettoyage du masque + extraction de la conjonctive (sans amélioration des couleurs)")
                st.write(f"**{t('tech_decision')}:** {t('tech_decision_value')}")

            # تحذير طبي
            st.markdown(f"""
            <div class="disclaimer">
                ⚠️ <strong>{t('disclaimer')}</strong><br>
                {t('disclaimer_text')}<br>
                {t('disclaimer_consult')}
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
<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:1.8rem 0;">
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

# ========== FOOTER ==========
st.markdown(f"""
<div class="app-footer">
    <p>{t('footer_text')}</p>
    <p style="font-size:12px; margin-top:4px;">
        <a href="#">{t('nav_about')}</a> • 
        <a href="#">{t('sidebar_health_title')}</a> • 
        <a href="#">{t('sidebar_doctor_title')}</a>
    </p>
</div>
""", unsafe_allow_html=True)
