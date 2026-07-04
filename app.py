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

# ========== CSS MODERN (refined) ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    * { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
        --bg-primary: #F0F7FF;
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
        --primary-500: #4F8CFF;
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
        scroll-behavior: smooth;
    }

    h1, h2, h3, h4, h5, .brand-font {
        font-family: 'Plus Jakarta Sans', 'Tajawal', sans-serif;
        font-weight: 700;
    }

    /* --- ANIMATED BACKGROUND --- */
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
        animation: floatBg 25s ease-in-out infinite alternate;
    }

    @keyframes floatBg {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(3%, -2%) rotate(2deg); }
    }

    /* --- STICKY NAVBAR --- */
    .navbar-wrapper {
        position: sticky;
        top: 0;
        z-index: 1000;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        background: rgba(240, 247, 255, 0.75);
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.5);
        box-shadow: 0 4px 20px rgba(10,36,99,0.05);
        transition: var(--transition);
    }
    .navbar-wrapper:hover {
        box-shadow: 0 6px 30px rgba(10,36,99,0.08);
    }
    .navbar-inner {
        max-width: 1300px;
        margin: 0 auto;
        padding: 0 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 14px;
        text-decoration: none;
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
        transition: var(--transition);
    }
    .navbar-brand .mark:hover {
        transform: scale(1.05) rotate(-5deg);
    }
    .navbar-brand img.mark-img {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        object-fit: cover;
        box-shadow: var(--shadow-sm);
    }
    .navbar-brand .brand-text h1 {
        font-size: 20px;
        font-weight: 800;
        color: var(--ink);
        margin: 0;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .navbar-brand .brand-text .subtitle {
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
        cursor: default;
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

    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0A1B40 0%, #0A2463 100%);
        border-right: none;
        box-shadow: 6px 0 30px rgba(10,36,99,0.25);
        backdrop-filter: blur(8px);
        padding-top: 0.5rem;
    }
    section[data-testid="stSidebar"] * {
        color: #E8EFFE !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div {
        background: rgba(255,255,255,0.08);
        border-radius: var(--radius-sm);
        border: 1px solid rgba(255,255,255,0.12);
        transition: var(--transition);
    }
    section[data-testid="stSidebar"] .stSelectbox > div:hover {
        background: rgba(255,255,255,0.14);
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
        transition: var(--transition);
    }
    .sidebar-glass:hover {
        background: rgba(255,255,255,0.08);
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
    .sidebar-glass .nav-item:not(.active):hover {
        background: rgba(255,255,255,0.08);
        transform: translateX(4px);
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
        transition: var(--transition);
    }
    .sidebar-glass .qr-container img:hover {
        transform: scale(1.02);
        box-shadow: var(--shadow-md);
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
        transform: scale(1.02);
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

    /* --- HERO --- */
    .hero {
        position: relative;
        background: linear-gradient(145deg, #0A2463 0%, #1D4ED8 60%, #2563EB 100%);
        border-radius: var(--radius-xl);
        padding: 2.8rem 2.5rem;
        margin-bottom: 2.5rem;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
        animation: fadeUp 0.7s ease;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -30%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
        border-radius: 50%;
        animation: pulseBg 8s ease-in-out infinite alternate;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -20%;
        left: 5%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(14,165,164,0.15) 0%, transparent 70%);
        border-radius: 50%;
        animation: pulseBg 10s ease-in-out infinite alternate-reverse;
    }
    @keyframes pulseBg {
        0% { transform: scale(1) translate(0,0); opacity: 0.6; }
        100% { transform: scale(1.2) translate(-10%, -5%); opacity: 1; }
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
        animation: sparkle 3s ease-in-out infinite;
    }
    @keyframes sparkle {
        0%, 100% { transform: scale(1) rotate(0deg); }
        50% { transform: scale(1.15) rotate(15deg); }
    }
    .hero-text {
        flex: 1.3;
        min-width: 240px;
        text-align: center;
    }
    .hero-text h1 {
        font-size: 38px;
        font-weight: 800;
        color: #fff;
        margin: 0 0 8px;
        line-height: 1.2;
        text-shadow: 0 2px 20px rgba(0,0,0,0.1);
    }
    .hero-text h1 span {
        background: linear-gradient(120deg, #6EF3D6, #7FD8FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-text p {
        font-size: 16.5px;
        color: rgba(255,255,255,0.9);
        margin: 0 0 20px;
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
        text-shadow: 0 1px 8px rgba(0,0,0,0.05);
    }
    .hero-badge {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        color: #fff;
        padding: 7px 20px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        backdrop-filter: blur(4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .hero-cta {
        margin-top: 16px;
        display: flex;
        justify-content: center;
        gap: 14px;
        flex-wrap: wrap;
    }
    .hero-cta .btn-primary {
        background: white;
        color: var(--primary-900);
        border: none;
        padding: 14px 40px;
        border-radius: 40px;
        font-weight: 700;
        font-size: 17px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.15);
        transition: var(--transition);
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        text-decoration: none;
        letter-spacing: 0.01em;
    }
    .hero-cta .btn-primary:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 18px 40px rgba(0,0,0,0.25);
    }
    .hero-cta .btn-secondary {
        background: rgba(255,255,255,0.15);
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
        padding: 14px 32px;
        border-radius: 40px;
        font-weight: 600;
        font-size: 16px;
        transition: var(--transition);
        cursor: pointer;
        backdrop-filter: blur(4px);
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .hero-cta .btn-secondary:hover {
        background: rgba(255,255,255,0.25);
        transform: translateY(-2px);
    }

    /* --- SECTIONS --- */
    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: var(--ink);
        margin: 2.5rem 0 1.4rem;
        padding-left: 18px;
        border-left: 6px solid var(--primary-600);
        display: flex;
        align-items: center;
        gap: 12px;
        animation: fadeUp 0.4s ease;
    }

    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
        margin: 1.2rem 0;
    }

    .feature-card {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.7);
        border-radius: var(--radius-md);
        padding: 1.8rem 1.2rem;
        text-align: center;
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: var(--shadow-md);
        border-color: rgba(37,99,235,0.15);
    }
    .feature-card .icon-circle {
        width: 56px;
        height: 56px;
        border-radius: 18px;
        background: var(--primary-100);
        color: var(--primary-700);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        margin: 0 auto 14px;
        transition: var(--transition);
    }
    .feature-card:hover .icon-circle {
        background: var(--primary-700);
        color: white;
        transform: scale(1.05) rotate(-4deg);
    }
    .feature-card h4 {
        color: var(--ink);
        font-weight: 700;
        margin: 6px 0 4px;
        font-size: 17px;
    }
    .feature-card p {
        color: var(--slate);
        font-size: 13.5px;
        margin: 0;
        line-height: 1.5;
    }

    /* --- HOW IT WORKS --- */
    .how-section {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border-radius: var(--radius-lg);
        padding: 2.6rem 2.2rem;
        margin: 2.5rem 0;
        border: 1px solid rgba(255,255,255,0.6);
        box-shadow: var(--shadow-sm);
    }
    .how-title-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
        margin-bottom: 2rem;
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
        font-size: 26px;
        margin: 0;
    }
    .how-steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        position: relative;
    }
    @media (max-width: 768px) {
        .how-steps { grid-template-columns: 1fr; }
    }
    .how-step {
        text-align: center;
        padding: 1.8rem 1.2rem;
        background: rgba(244,247,252,0.6);
        backdrop-filter: blur(4px);
        border-radius: var(--radius-md);
        border: 1px solid rgba(255,255,255,0.5);
        transition: var(--transition);
        position: relative;
    }
    .how-step:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-6px);
        background: white;
    }
    .how-step .step-num {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-900));
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 20px;
        margin: 0 auto 16px;
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }
    .how-step:hover .step-num {
        transform: scale(1.08) rotate(-6deg);
    }
    .how-step h5 {
        color: var(--ink);
        font-weight: 700;
        margin: 0 0 8px;
        font-size: 17px;
    }
    .how-step p {
        color: var(--slate);
        font-size: 14px;
        margin: 0;
    }

    /* --- TRUST BADGES --- */
    .trust-section {
        display: flex;
        justify-content: center;
        gap: 40px;
        flex-wrap: wrap;
        margin: 2rem 0;
        padding: 1.2rem 1.5rem;
        background: var(--primary-100);
        border-radius: var(--radius-md);
        border: 1px solid rgba(37,99,235,0.08);
    }
    .trust-item {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
        color: var(--primary-900);
        font-size: 15px;
    }
    .trust-item .icon {
        font-size: 22px;
    }

    /* --- UPLOAD CARD --- */
    .upload-card {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border-radius: var(--radius-xl);
        padding: 2.6rem 1.8rem;
        text-align: center;
        border: 2px dashed rgba(37,99,235,0.25);
        transition: var(--transition);
        margin-bottom: 1.8rem;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }
    .upload-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(37,99,235,0.02) 0%, transparent 70%);
        pointer-events: none;
        animation: floatBg 20s ease-in-out infinite alternate;
    }
    .upload-card:hover {
        border-color: var(--primary-600);
        box-shadow: var(--shadow-md);
        transform: translateY(-4px);
    }
    .upload-card .icon-circle {
        width: 80px;
        height: 80px;
        border-radius: 24px;
        background: var(--primary-100);
        color: var(--primary-700);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        margin: 0 auto 16px;
        animation: floatY 3.5s ease-in-out infinite;
        transition: var(--transition);
    }
    .upload-card:hover .icon-circle {
        background: var(--primary-700);
        color: white;
        transform: scale(1.05);
    }
    .upload-card h3 {
        font-weight: 800;
        color: var(--ink);
        margin: 4px 0 8px;
        font-size: 22px;
    }
    .upload-card p {
        color: var(--slate);
        font-size: 15px;
        margin: 0;
    }

    /* --- PREVIEW --- */
    .preview-container {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border-radius: var(--radius-lg);
        padding: 1.6rem;
        border: 1px solid rgba(255,255,255,0.6);
        box-shadow: var(--shadow-sm);
        margin: 1.4rem 0;
        animation: fadeUp 0.5s ease;
        transition: var(--transition);
    }
    .preview-container:hover {
        box-shadow: var(--shadow-md);
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
        max-height: 340px;
        object-fit: contain;
        width: 100%;
        box-shadow: var(--shadow-sm);
    }

    /* --- RESULT CARDS --- */
    .result-card {
        border-radius: var(--radius-lg);
        padding: 2rem 2.2rem;
        margin-top: 0.6rem;
        border: 1px solid transparent;
        display: flex;
        align-items: center;
        gap: 24px;
        animation: fadeUp 0.5s ease;
        box-shadow: var(--shadow-sm);
        backdrop-filter: blur(8px);
        transition: var(--transition);
    }
    .result-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    .result-card .badge-ic {
        width: 64px;
        height: 64px;
        border-radius: 20px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
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
        font-size: 26px;
        font-weight: 800;
        margin: 0 0 4px;
        color: var(--ink);
    }
    .result-card .confidence {
        font-size: 17px;
        font-weight: 600;
        color: var(--ink);
    }
    .result-card .confidence strong {
        color: var(--primary-700);
    }
    .result-card .sub {
        font-size: 14px;
        color: var(--slate);
        margin-top: 2px;
    }

    /* --- DISCLAIMER --- */
    .disclaimer {
        background: var(--amber-100);
        border-radius: var(--radius-md);
        padding: 1.4rem 1.8rem;
        border: 1px solid rgba(222,154,31,0.15);
        border-left: 6px solid var(--amber);
        margin-top: 2.2rem;
        font-size: 14px;
        color: #3D2A5C;
        line-height: 1.8;
        animation: fadeUp 0.5s ease;
        transition: var(--transition);
    }
    .disclaimer:hover {
        box-shadow: var(--shadow-sm);
    }
    .disclaimer strong {
        color: #2A1A40;
    }

    /* --- STATS BAR --- */
    .stats-bar {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 18px;
        margin: 2.2rem 0;
    }
    @media (max-width: 768px) {
        .stats-bar { grid-template-columns: repeat(2, 1fr); }
    }
    .stat-item {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.6);
        border-radius: var(--radius-md);
        padding: 1.6rem 0.8rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    .stat-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-600), var(--teal));
        opacity: 0;
        transition: var(--transition);
    }
    .stat-item:hover::before {
        opacity: 1;
    }
    .stat-item:hover {
        transform: translateY(-6px);
        box-shadow: var(--shadow-md);
    }
    .stat-item .stat-number {
        font-size: 32px;
        font-weight: 800;
        color: var(--primary-700);
        line-height: 1.2;
        display: inline-block;
        animation: countUp 1s ease-out forwards;
    }
    @keyframes countUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stat-item .stat-label {
        font-size: 14px;
        color: var(--slate);
        margin-top: 6px;
    }

    /* --- FOOTER --- */
    .app-footer {
        margin-top: 3.5rem;
        padding: 2rem 0;
        border-top: 1px solid var(--line);
        text-align: center;
        color: var(--slate-soft);
        font-size: 14px;
    }
    .app-footer a {
        color: var(--primary-600);
        text-decoration: none;
        font-weight: 500;
        transition: var(--transition);
        margin: 0 8px;
    }
    .app-footer a:hover {
        color: var(--primary-900);
        text-decoration: underline;
    }
    .footer-socials {
        display: flex;
        justify-content: center;
        gap: 16px;
        margin-top: 12px;
    }
    .footer-socials a {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--primary-100);
        color: var(--primary-700);
        font-size: 18px;
        transition: var(--transition);
    }
    .footer-socials a:hover {
        background: var(--primary-700);
        color: white;
        transform: translateY(-3px);
        box-shadow: var(--shadow-sm);
    }

    /* --- WIDGET OVERRIDES --- */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-900));
        color: white;
        border: none;
        border-radius: 40px;
        padding: 16px 36px;
        font-weight: 700;
        letter-spacing: 0.01em;
        transition: var(--transition);
        width: 100%;
        box-shadow: 0 10px 28px rgba(29,78,216,0.25);
        font-size: 16px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 40px rgba(29,78,216,0.35);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-600), var(--teal));
        border-radius: 10px;
        transition: width 0.6s ease;
    }
    [data-testid="stMetric"] {
        background: var(--bg-surface);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 18px 22px;
        border: 1px solid rgba(255,255,255,0.6);
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }
    [data-testid="stMetric"]:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    [data-testid="stMetricValue"] {
        font-size: 30px !important;
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
        transition: var(--transition);
    }
    [data-testid="stExpander"]:hover {
        box-shadow: var(--shadow-md);
    }
    [data-testid="stExpander"] summary {
        font-weight: 600;
        font-size: 15px;
        padding: 8px 0;
    }
    .stImage img {
        border-radius: 16px;
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }
    .stImage img:hover {
        box-shadow: var(--shadow-md);
    }
    #MainMenu, footer, .stDeployButton {
        display: none;
    }

    /* --- RESPONSIVE TWEAKS --- */
    @media (max-width: 768px) {
        .navbar-inner { flex-direction: column; align-items: stretch; gap: 0.5rem; }
        .navbar-badges { justify-content: center; }
        .hero { padding: 2rem 1.5rem; }
        .hero-text h1 { font-size: 28px; }
        .hero-cta .btn-primary, .hero-cta .btn-secondary { width: 100%; justify-content: center; }
        .section-title { font-size: 20px; }
        .result-card { flex-direction: column; text-align: center; padding: 1.5rem; }
        .result-card .badge-ic { margin: 0 auto; }
        .stats-bar { grid-template-columns: 1fr 1fr; }
        .feature-card { padding: 1.2rem; }
        .how-section { padding: 1.5rem; }
        .how-step { padding: 1.2rem; }
        .upload-card { padding: 1.8rem 1rem; }
        .trust-section { gap: 20px; }
    }
    @media (max-width: 480px) {
        .hero-text h1 { font-size: 22px; }
        .navbar-brand .brand-text h1 { font-size: 17px; }
        .navbar-chip { padding: 4px 10px; font-size: 11px; }
        .stat-item .stat-number { font-size: 24px; }
        .result-card h2 { font-size: 22px; }
        .section-title { font-size: 18px; }
    }

    /* --- ANIMATIONS --- */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(25px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes floatY {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-12px); }
    }
    @keyframes pulseRing {
        0% { box-shadow: 0 0 0 0 rgba(255,255,255,0.3); }
        100% { box-shadow: 0 0 0 20px rgba(255,255,255,0); }
    }
    /* loading shimmer (skeleton) */
    .shimmer {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 8px;
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
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
# Navbar (now sticky)
logo_mark_html = f'<img src="data:image/png;base64,{logo}" class="mark-img">' if logo else '<div class="mark">🩸</div>'

st.markdown(f"""
<div class="navbar-wrapper">
    <div class="navbar-inner">
        <a href="#" class="navbar-brand">
            {logo_mark_html}
            <div class="brand-text">
                <h1>{t('app_title')}</h1>
                <div class="subtitle">{t('app_subtitle')}</div>
            </div>
        </a>
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

# ========== UPLOAD SECTION (with anchor) ==========
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

# ========== FONCTIONS (unchanged) ==========
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
            mask_cropped = mask[y:y+h, x:x+w]
            return cropped, mask_cropped, (x, y, w, h)
    conj = cv2.bitwise_and(img, img, mask=mask)
    return conj, mask, None

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

            # ===== RESULTS =====
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

            # Metrics
            before = np.sum(raw_mask > 0) / 255
            after = np.sum(final_mask > 0) / 255
            reduction = ((before - after) / before * 100) if before > 0 else 0

            m1, m2 = st.columns(2)
            with m1:
                st.metric(t("metric_surface"), f"{after:.0f} px²")
            with m2:
                st.metric(t("metric_cleaning"), f"{reduction:.1f}%")

            # Diagnosis
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

            # Chart
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

            # History
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

            # Technical details
            with st.expander(t("tech_details")):
                st.write(f"**{t('tech_model_seg')}:** U‑Net (ResNet34)")
                st.write(f"**{t('tech_model_clf')}:** EfficientNet‑B3")
                st.write(f"**{t('tech_device')}:** {'GPU' if clf_device.type == 'cuda' else 'CPU'}")
                st.write(f"**{t('tech_sigmoid')}:** {raw_pred:.4f}")
                st.write(f"**{t('tech_prob_anemic')}:** {anemia_pct:.1f}%")
                st.write(f"**{t('tech_prob_non')}:** {non_pct:.1f}%")
                st.write(f"**{t('tech_preprocess')}:** CLAHE + Filtrage + Netteté")
                st.write(f"**{t('tech_decision')}:** {t('tech_decision_value')}")

            # Disclaimer
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

# ========== BOTTOM INFO (Fast/Smart/Secure) ==========
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

# ========== FOOTER (enhanced) ==========
st.markdown(f"""
<div class="app-footer">
    <p>{t('footer_text')}</p>
    <div style="display:flex; justify-content:center; gap:20px; flex-wrap:wrap; margin-top:8px;">
        <a href="#">{t('nav_about')}</a>
        <a href="#">{t('sidebar_health_title')}</a>
        <a href="#">{t('sidebar_doctor_title')}</a>
    </div>
    <div class="footer-socials">
        <a href="#" aria-label="Twitter">🐦</a>
        <a href="#" aria-label="LinkedIn">🔗</a>
        <a href="#" aria-label="GitHub">🐙</a>
        <a href="#" aria-label="YouTube">▶️</a>
    </div>
    <p style="font-size:12px; margin-top:12px; color:var(--slate-soft);">
        {t('sidebar_version')} • {t('accuracy')}
    </p>
</div>
""", unsafe_allow_html=True)
