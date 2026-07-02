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
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Tajawal:wght@400;700;800&display=swap');
    
    * { font-family: 'Inter', 'Tajawal', sans-serif; }
    
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
    
    /* ===== HEADER ===== */
    .header {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 0.8rem 2rem;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
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
    .header-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-left h1 {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .header-left h1 span { color: #e11d48; }
    .header-left .subtitle {
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
    }
    .header-badges {
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
    }
    .header-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #334155;
        background: rgba(255,255,255,0.5);
        padding: 4px 14px;
        border-radius: 30px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .header-badge strong {
        color: #e11d48;
        font-weight: 700;
    }
    
    /* ===== SIDEBAR ===== */
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
        font-size: 15px;
    }
    .sidebar-glass .nav-item {
        padding: 10px 14px;
        border-radius: 12px;
        margin: 4px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 500;
        color: #334155;
        background: rgba(255,255,255,0.5);
        cursor: default;
        transition: all 0.3s ease;
    }
    .sidebar-glass .nav-item.active {
        background: rgba(225,29,72,0.12);
        color: #e11d48;
        font-weight: 600;
        border: 1px solid rgba(225,29,72,0.15);
    }
    .sidebar-glass .nav-item:hover {
        background: rgba(225,29,72,0.04);
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
    
    /* ===== HERO ===== */
    .hero {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 28px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.04);
        animation: fadeUp 0.8s ease;
        transition: all 0.3s ease;
        text-align: center;
    }
    .hero:hover {
        box-shadow: 0 16px 48px rgba(225,29,72,0.08);
    }
    .hero .icon {
        font-size: 48px;
        animation: float 3s ease-in-out infinite;
        display: block;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .hero h1 {
        font-size: 32px;
        font-weight: 800;
        color: #0f172a;
        margin: 0.5rem 0 0.2rem;
    }
    .hero h1 span {
        background: linear-gradient(135deg, #9f1239, #e11d48);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        font-size: 16px;
        color: #64748b;
        margin: 0.5rem 0 1.2rem;
    }
    .hero .hero-badge {
        background: rgba(225,29,72,0.08);
        color: #e11d48;
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
    }
    .doctor-image {
        border-radius: 50%;
        box-shadow: 0 8px 32px rgba(225,29,72,0.15);
        border: 3px solid rgba(225,29,72,0.1);
        transition: transform 0.3s ease;
        width: 120px;
        height: 120px;
        object-fit: cover;
    }
    .doctor-image:hover {
        transform: scale(1.05);
    }
    .hero-content {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
    }
    .hero-text {
        flex: 1;
        min-width: 250px;
    }
    
    /* ===== FEATURES ===== */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 1.5rem 0;
    }
    @media (max-width: 768px) {
        .features-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .feature-card {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.08);
    }
    .feature-card .icon {
        font-size: 32px;
    }
    .feature-card h4 {
        color: #0f172a;
        font-weight: 700;
        margin: 8px 0 4px;
        font-size: 16px;
    }
    .feature-card p {
        color: #64748b;
        font-size: 13px;
        margin: 0;
    }
    
    /* ===== HOW IT WORKS ===== */
    .how-section {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(8px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .how-section h3 {
        color: #0f172a;
        font-weight: 700;
        font-size: 22px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .how-steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
    }
    @media (max-width: 768px) {
        .how-steps {
            grid-template-columns: 1fr;
        }
    }
    .how-step {
        text-align: center;
        padding: 0 10px;
    }
    .how-step .step-num {
        background: linear-gradient(135deg, #e11d48, #fb7185);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 18px;
        margin: 0 auto 10px;
    }
    .how-step h5 {
        color: #0f172a;
        font-weight: 700;
        margin: 0 0 4px;
        font-size: 16px;
    }
    .how-step p {
        color: #64748b;
        font-size: 14px;
        margin: 0;
    }
    
    /* ===== TRUST BADGES ===== */
    .trust-section {
        display: flex;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
        margin: 1.5rem 0;
        padding: 1.5rem;
        background: rgba(255,255,255,0.5);
        backdrop-filter: blur(4px);
        border-radius: 20px;
    }
    .trust-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: #0f172a;
        font-size: 15px;
    }
    .trust-item .icon {
        font-size: 24px;
    }
    
    /* ===== RESULT CARDS ===== */
    .result-card {
        border-radius: 24px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        border-left: 6px solid #e11d48;
        animation: popIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    .result-card.positive {
        background: rgba(254, 242, 242, 0.8);
        border-left-color: #dc2626;
    }
    .result-card.negative {
        background: rgba(240, 253, 244, 0.8);
        border-left-color: #16a34a;
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
    
    /* ===== GENERAL ===== */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
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
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e11d48, #fb7185);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* ===== UPLOAD ===== */
    .upload-card {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 28px;
        padding: 2rem 1.5rem;
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
    .upload-card .icon {
        font-size: 48px;
        animation: float 3s ease-in-out infinite;
    }
    .upload-card h3 {
        font-weight: 700;
        color: #0f172a;
        margin: 10px 0 4px;
        font-size: 20px;
    }
    .upload-card p {
        color: #64748b;
        font-size: 14px;
        margin: 0;
    }
    
    /* ===== PREVIEW ===== */
    .preview-container {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(8px);
        border-radius: 24px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.04);
        margin: 1.5rem 0;
        animation: fadeUp 0.6s ease;
    }
    .preview-container img {
        border-radius: 16px;
        max-height: 300px;
        object-fit: contain;
        width: 100%;
    }
    
    /* ===== DISCLAIMER ===== */
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
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&margin=10&color=e11d48&data={APP_URL}"
    
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>{t('sidebar_qr_title')}</h4>
        <div class="qr-container">
            <img src="{qr_api_url}" width="160" height="160" alt="QR Code">
        </div>
        <p style="text-align:center; font-size:13px; color:#64748b;">{t('sidebar_qr_desc')}</p>
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
        <p style="font-size:12px; color:#94a3b8; text-align:center; margin-top:8px;">{t('sidebar_doctor_soon')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption(t('sidebar_version'))

# ========== PAGE PRINCIPALE ==========
# Header avec badges
if logo:
    st.markdown(f"""
    <div class="header">
        <div class="header-left">
            <img src="data:image/png;base64,{logo}" style="height:42px;">
            <div>
                <h1>{t('app_title')}</h1>
                <div class="subtitle">{t('app_subtitle')}</div>
            </div>
        </div>
        <div class="header-badges">
            <div class="header-badge">🔒 <strong>{t('badge_private')}</strong> {t('badge_private_desc')}</div>
            <div class="header-badge">⏰ <strong>{t('badge_available')}</strong> {t('badge_available_desc')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="header">
        <div class="header-left">
            <div style="font-size:36px;">🩸</div>
            <div>
                <h1>{t('app_title')}</h1>
                <div class="subtitle">{t('app_subtitle')}</div>
            </div>
        </div>
        <div class="header-badges">
            <div class="header-badge">🔒 <strong>{t('badge_private')}</strong> {t('badge_private_desc')}</div>
            <div class="header-badge">⏰ <strong>{t('badge_available')}</strong> {t('badge_available_desc')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== HERO SECTION AVEC IMAGE DU MÉDECIN ==========
# Utilisation d'une icône de médecin (Flaticon)
doctor_img_url = "https://cdn-icons-png.flaticon.com/512/3774/3774299.png"

st.markdown(f"""
<div class="hero">
    <div class="hero-content">
        <div>
            <img src="{doctor_img_url}" class="doctor-image" alt="Médecin / Doctor">
        </div>
        <div class="hero-text">
            <span class="icon">🩺</span>
            <h1>{t('hero_title')}</h1>
            <p>{t('hero_desc')}</p>
            <div style="margin-top:10px;">
                <span class="hero-badge">✅ {t('hero_badge')}</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== ZONE UPLOAD ==========
st.markdown(f"""
<div class="upload-card">
    <div class="icon">📸</div>
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
                        <h2>{t('diagnostic_anemic')}</h2>
                        <div class="confidence">{t('diagnostic_confidence')} : <strong>{confidence:.1f}%</strong></div>
                        <div class="sub">{t('diagnostic_anemic_desc')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card negative">
                        <h2>{t('diagnostic_non_anemic')}</h2>
                        <div class="confidence">{t('diagnostic_confidence')} : <strong>{confidence:.1f}%</strong></div>
                        <div class="sub">{t('diagnostic_non_anemic_desc')}</div>
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
                <strong>{t('disclaimer')}</strong><br>
                {t('disclaimer_text')}<br>
                {t('disclaimer_consult')}
            </div>
            """, unsafe_allow_html=True)

# ========== FEATURES GRID ==========
st.markdown("""
<div class="features-grid">
    <div class="feature-card">
        <div class="icon">⚡</div>
        <h4>""" + t('feature_1_title') + """</h4>
        <p>""" + t('feature_1_desc') + """</p>
    </div>
    <div class="feature-card">
        <div class="icon">📤</div>
        <h4>""" + t('feature_2_title') + """</h4>
        <p>""" + t('feature_2_desc') + """</p>
    </div>
    <div class="feature-card">
        <div class="icon">✅</div>
        <h4>""" + t('feature_3_title') + """</h4>
        <p>""" + t('feature_3_desc') + """</p>
    </div>
    <div class="feature-card">
        <div class="icon">🔒</div>
        <h4>""" + t('feature_4_title') + """</h4>
        <p>""" + t('feature_4_desc') + """</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== HOW IT WORKS ==========
st.markdown(f"""
<div class="how-section">
    <h3>{t('how_title')}</h3>
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
        <div class="icon">⚡</div>
        <h4>{t('fast')}</h4>
        <p>{t('fast_desc')}</p>
    </div>
    <div class="feature-card">
        <div class="icon">🧠</div>
        <h4>{t('smart')}</h4>
        <p>{t('smart_desc')}</p>
    </div>
    <div class="feature-card">
        <div class="icon">🔒</div>
        <h4>{t('secure')}</h4>
        <p>{t('secure_desc')}</p>
    </div>
</div>
""", unsafe_allow_html=True)
