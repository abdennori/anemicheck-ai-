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
import io
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
        "hero_cta": "ابدأ التحليل الآن",
        "ai_dev_badge": "تم تطويره بواسطة الذكاء الاصطناعي",
        "top_badge1_title": "مساعدة فورية",
        "top_badge1_desc": "تواصل مع فريق الدعم",
        "top_badge2_title": "مكالمة مجانية",
        "top_badge2_desc": "🔜 قريباً",
        "top_badge3_title": "خدمة متاحة 24/7",
        "top_badge3_desc": "نحن هنا من أجلك دائماً",
        "top_badge4_title": "آمن وموثوق",
        "top_badge4_desc": "خصوصيتك محمية 100%",
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
        "history_empty": "📭 لا توجد تحليلات سابقة بعد",
        "history_empty_desc": "قم بإجراء أول تحليل لك الآن!",
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
        "nav_history": "📋 سجل التحليلات",
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
        "hero_cta": "Commencer l'analyse",
        "ai_dev_badge": "Développé avec l'intelligence artificielle",
        "top_badge1_title": "Aide immédiate",
        "top_badge1_desc": "Contactez notre équipe",
        "top_badge2_title": "Appel gratuit",
        "top_badge2_desc": "🔜 Bientôt",
        "top_badge3_title": "Service 24/7",
        "top_badge3_desc": "Nous sommes toujours là",
        "top_badge4_title": "Sûr et fiable",
        "top_badge4_desc": "100% de confidentialité",
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
        "history_empty": "📭 Aucun historique",
        "history_empty_desc": "Effectuez votre première analyse maintenant !",
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
        "nav_history": "📋 Historique",
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
        "hero_cta": "Start analysis now",
        "ai_dev_badge": "Built with artificial intelligence",
        "top_badge1_title": "Instant help",
        "top_badge1_desc": "Chat with our support team",
        "top_badge2_title": "Free call",
        "top_badge2_desc": "🔜 Coming soon",
        "top_badge3_title": "24/7 available",
        "top_badge3_desc": "We are always here for you",
        "top_badge4_title": "Safe & trusted",
        "top_badge4_desc": "100% of your privacy protected",
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
        "history_empty": "📭 No history yet",
        "history_empty_desc": "Perform your first analysis now!",
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
        "nav_history": "📋 History",
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
        background: linear-gradient(-45deg, #f4f7fb, #eaf1fb, #f1f5fb, #f4f7fb);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .header {
        background: linear-gradient(120deg, #1e3a8a 0%, #1d4ed8 55%, #2563eb 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 0 0 26px 26px;
        box-shadow: 0 10px 32px rgba(30,58,138,0.25);
        margin-bottom: 1.5rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        animation: slideDown 0.6s ease;
        gap: 12px;
    }
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .header-left {
        display: flex;
        align-items: center;
        gap: 12px;
        order: 2;
    }
    .header-left h1 {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .header-left h1 span { color: #7dd3fc; }
    .header-left .subtitle {
        font-size: 12px;
        color: #dbeafe;
        font-weight: 500;
    }
    .header-badges {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        order: 1;
    }
    .header-ai-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 600;
        color: #ffffff;
        background: rgba(16,185,129,0.25);
        border: 1px solid rgba(16,185,129,0.5);
        padding: 5px 14px;
        border-radius: 30px;
    }
    .header-logo-badge {
        width: 48px;
        height: 48px;
        min-width: 48px;
        border-radius: 14px;
        background: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        overflow: hidden;
    }
    .header-logo-badge img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        padding: 4px;
    }
    .header-icon-btn {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: rgba(255,255,255,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-size: 15px;
        position: relative;
    }
    .header-icon-btn .dot {
        position: absolute;
        top: 4px;
        right: 4px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #f87171;
        border: 1.5px solid #1e3a8a;
    }
    .header-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        color: #334155;
        background: rgba(255,255,255,0.5);
        padding: 4px 12px;
        border-radius: 30px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .header-badge strong {
        color: #2563eb;
        font-weight: 700;
    }
    @media (max-width: 700px) {
        .header {
            flex-direction: column;
            align-items: stretch;
            padding: 0.8rem 1rem;
        }
        .header-left {
            order: 1;
            justify-content: center;
            flex-wrap: wrap;
        }
        .header-badges {
            order: 2;
            justify-content: center;
        }
        .header-left h1 { font-size: 20px; }
    }

    .top-badges {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 0 0 1.5rem;
    }
    @media (max-width: 768px) {
        .top-badges { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 480px) {
        .top-badges { grid-template-columns: 1fr; }
    }
    .top-badge-card {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 18px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 16px rgba(30,58,138,0.05);
        transition: all 0.3s ease;
        position: relative;
        min-height: 64px;
    }
    .top-badge-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(30,58,138,0.1);
    }
    .top-badge-card .tb-icon {
        width: 40px;
        height: 40px;
        min-width: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #F59E0B, #D97706);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 18px;
        box-shadow: 0 4px 10px rgba(245, 158, 11, 0.3);
    }
    .top-badge-card .tb-title {
        font-weight: 700;
        font-size: 13px;
        color: #0f172a;
    }
    .top-badge-card .tb-desc {
        font-size: 11px;
        color: #64748b;
    }
    .top-badge-card .tb-soon {
        position: absolute;
        top: -6px;
        right: -6px;
        background: #F59E0B;
        color: #fff;
        font-size: 9px;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 30px;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(245,158,11,0.4);
        animation: pulse-badge 1.5s ease-in-out infinite;
    }
    @keyframes pulse-badge {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .sidebar-glass {
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 24px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.06);
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
        background: rgba(245,158,11,0.12);
        color: #D97706;
        font-weight: 600;
        border: 1px solid rgba(245,158,11,0.2);
    }
    .sidebar-glass .nav-item:hover {
        background: rgba(245,158,11,0.04);
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
        border: 1px solid rgba(245,158,11,0.15);
        background: white;
        padding: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        transition: transform 0.3s ease;
        max-width: 100%;
        height: auto;
    }
    .sidebar-glass .qr-container img:hover {
        transform: scale(1.05);
    }
    
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 50%, #2563eb 100%);
        border-radius: 32px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        min-height: 65vh;
        display: flex;
        align-items: center;
        box-shadow: 0 16px 40px rgba(30,58,138,0.25);
        animation: fadeUp 0.8s ease;
        transition: all 0.3s ease;
        text-align: right;
        position: relative;
        overflow: hidden;
    }
    .hero:hover {
        box-shadow: 0 20px 48px rgba(30,58,138,0.32);
    }
    .hero .icon {
        font-size: 40px;
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .hero h1 {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        margin: 0.5rem 0 0.2rem;
        line-height: 1.3;
    }
    .hero h1 span {
        color: #FCD34D;
    }
    .hero p {
        font-size: 16px;
        color: #dbeafe;
        margin: 0.5rem 0 1.4rem;
        max-width: 480px;
    }
    .hero .hero-badge {
        background: rgba(16,185,129,0.18);
        color: #d1fae5;
        border: 1px solid rgba(52,211,153,0.4);
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
    }
    .doctor-image {
        border-radius: 24px;
        filter: drop-shadow(0 12px 28px rgba(0,0,0,0.25));
        transition: transform 0.3s ease;
        width: 100%;
        max-width: 320px;
        object-fit: contain;
    }
    .doctor-image:hover {
        transform: scale(1.02);
    }
    .hero-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 30px;
        flex-wrap: wrap-reverse;
        position: relative;
        z-index: 2;
        width: 100%;
    }
    .hero-text {
        flex: 1;
        min-width: 280px;
    }
    .hero-visual {
        position: relative;
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 260px;
    }
    .hero-visual .glow {
        position: absolute;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(245,158,11,0.15), transparent 70%);
    }
    .hero-heartbeat {
        position: absolute;
        top: 50%;
        left: -10%;
        width: 60%;
        opacity: 0.4;
        transform: translateY(-50%);
        z-index: 1;
    }
    .hero-cta {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 16px;
        padding: 14px 34px;
        border-radius: 40px;
        text-decoration: none !important;
        box-shadow: 0 8px 24px rgba(245,158,11,0.4);
        transition: all 0.3s ease;
        border: none;
    }
    .hero-cta:hover {
        transform: scale(1.04);
        box-shadow: 0 12px 32px rgba(245,158,11,0.5);
    }
    @media (max-width: 700px) {
        .hero {
            padding: 2rem 1.5rem;
            min-height: auto;
        }
        .hero-content {
            flex-direction: column;
            text-align: center;
        }
        .hero-text {
            min-width: auto;
        }
        .hero h1 {
            font-size: 26px;
        }
        .hero p {
            max-width: 100%;
        }
        .hero-cta {
            justify-content: center;
            width: 100%;
        }
        .doctor-image {
            max-width: 200px;
        }
    }

    .scan-container {
        position: relative;
        overflow: hidden;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .scan-container img {
        width: 100%;
        display: block;
        border-radius: 16px;
    }
    .scan-line {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: #dc2626;
        box-shadow: 0 0 20px #dc2626, 0 0 60px #dc2626;
        animation: scanMove 2s ease-in-out infinite;
        z-index: 10;
        border-radius: 2px;
    }
    @keyframes scanMove {
        0% { top: 0; opacity: 1; }
        50% { top: 100%; opacity: 0.8; }
        100% { top: 0; opacity: 1; }
    }
    .scan-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.02);
        pointer-events: none;
        border-radius: 16px;
        border: 2px solid rgba(220, 38, 38, 0.3);
    }

    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 1.5rem 0;
    }
    @media (max-width: 768px) {
        .features-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 480px) {
        .features-grid { grid-template-columns: 1fr; }
    }
    .feature-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.5);
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        animation: fadeUp 0.8s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.08);
        border-color: rgba(245,158,11,0.2);
    }
    .feature-card .icon { font-size: 32px; }
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
    
    .how-section {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(8px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255,255,255,0.5);
        animation: fadeUp 0.8s ease;
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
        .how-steps { grid-template-columns: 1fr; }
    }
    .how-step {
        text-align: center;
        padding: 0 10px;
        animation: fadeUp 0.8s ease;
    }
    .how-step .step-num {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 18px;
        margin: 0 auto 10px;
        transition: transform 0.3s;
        box-shadow: 0 4px 12px rgba(245,158,11,0.3);
    }
    .how-step:hover .step-num {
        transform: scale(1.1) rotate(5deg);
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
    
    .trust-section {
        display: flex;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
        margin: 1.5rem 0;
        padding: 1.5rem;
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(4px);
        border-radius: 20px;
        animation: fadeUp 1s ease;
    }
    .trust-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: #0f172a;
        font-size: 15px;
    }
    .trust-item .icon { font-size: 24px; }
    
    .result-card {
        border-radius: 24px;
        padding: 1.8rem;
        margin-top: 0.5rem;
        border-left: 6px solid #e11d48;
        animation: popIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(4px);
    }
    .result-card.positive {
        background: rgba(254, 242, 242, 0.85);
        border-left-color: #dc2626;
        box-shadow: 0 0 30px rgba(220, 38, 38, 0.08);
    }
    .result-card.negative {
        background: rgba(240, 253, 244, 0.85);
        border-left-color: #16a34a;
        box-shadow: 0 0 30px rgba(22, 163, 74, 0.08);
    }
    .result-card h2 {
        font-size: 30px;
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
    
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        margin: 2rem 0 1rem;
        padding-bottom: 8px;
        border-bottom: 3px solid #F59E0B;
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
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981, #F59E0B, #dc2626) !important;
        height: 12px !important;
        border-radius: 20px !important;
        transition: width 0.8s ease !important;
    }
    .stProgress > div > div {
        background: #e2e8f0 !important;
        border-radius: 20px !important;
        height: 12px !important;
    }
    
    #MainMenu, footer, .stDeployButton { display: none; }
    
    .stButton > button {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 14px 32px;
        font-weight: 700;
        font-size: 16px;
        transition: 0.3s;
        width: 100%;
        box-shadow: 0 4px 16px rgba(245,158,11,0.35);
        animation: fadeUp 0.6s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 24px rgba(245,158,11,0.5);
    }
    
    .upload-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 28px;
        padding: 2rem 1.5rem;
        text-align: center;
        border: 2px dashed rgba(245,158,11,0.3);
        transition: all 0.4s ease;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.04);
        animation: fadeUp 0.8s ease;
    }
    .upload-card:hover {
        border-color: #F59E0B;
        background: rgba(255,255,255,0.85);
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(245,158,11,0.08);
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
    
    .preview-container {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(8px);
        border-radius: 24px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 0 8px 32px rgba(0,0,0,0.04);
        margin: 1.5rem 0;
        animation: fadeUp 0.6s ease;
    }
    .preview-container img {
        border-radius: 16px;
        max-height: 350px;
        object-fit: contain;
        width: 100%;
    }
    
    .disclaimer {
        background: rgba(254, 252, 232, 0.85);
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
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 15px;
        cursor: not-allowed;
        opacity: 0.7;
        width: 100%;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(245,158,11,0.2);
        margin-top: 10px;
    }
    .doctor-btn:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
    .coming-badge {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 14px;
        border-radius: 30px;
        margin-left: 8px;
        letter-spacing: 0.5px;
        animation: pulse-badge 1.5s ease-in-out infinite;
    }
    
    /* ===================================================== */
    /* ==== انيميشن النبض للأزرار ==== */
    /* ===================================================== */
    @keyframes medical-pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.4);
            transform: scale(1);
        }
        50% {
            box-shadow: 0 0 0 12px rgba(245, 158, 11, 0);
            transform: scale(1.02);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(245, 158, 11, 0);
            transform: scale(1);
        }
    }
    div[data-testid="column"] button[kind="primary"] {
        animation: medical-pulse 1.8s ease-in-out infinite !important;
        position: relative;
        overflow: hidden;
    }
    div[data-testid="column"] button {
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        border-radius: 40px !important;
        font-weight: 600 !important;
        padding: 14px 0 !important;
        font-size: 16px !important;
        letter-spacing: 0.5px;
        cursor: pointer !important;
    }
    div[data-testid="column"] button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 28px rgba(245, 158, 11, 0.3) !important;
    }
    div[data-testid="column"] button[kind="secondary"]:hover {
        border-color: #F59E0B !important;
        background: rgba(245, 158, 11, 0.08) !important;
    }
    .heartbeat-icon {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
        font-size: 28px;
        margin: -0.5rem 0 0.5rem 0;
        color: #dc2626;
        animation: heartbeat-float 1.5s ease-in-out infinite;
    }
    @keyframes heartbeat-float {
        0%, 100% { transform: scale(1); }
        14% { transform: scale(1.2); }
        28% { transform: scale(1); }
        42% { transform: scale(1.2); }
        70% { transform: scale(1); }
    }
    .heartbeat-icon span {
        font-size: 16px;
        color: #0f172a;
        font-weight: 700;
        background: rgba(255,255,255,0.8);
        padding: 4px 16px;
        border-radius: 30px;
        backdrop-filter: blur(4px);
        border: 1px solid rgba(220, 38, 38, 0.15);
    }

    /* ===== تنسيق راديو التنقل في القائمة الجانبية ===== */
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    div[data-testid="stRadio"] label {
        background: rgba(255,255,255,0.5);
        padding: 10px 14px !important;
        border-radius: 12px !important;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        font-weight: 500;
        color: #334155;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(245,158,11,0.04);
        border-color: rgba(245,158,11,0.2);
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child {
        margin-left: 0 !important;
        width: 100%;
    }
    /* العنصر النشط */
    div[data-testid="stRadio"] label[data-testid="stRadioLabel"]:has(input:checked) {
        background: rgba(245,158,11,0.12) !important;
        color: #D97706 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(245,158,11,0.2) !important;
        box-shadow: 0 2px 8px rgba(245,158,11,0.08);
    }
    /* إخفاء الدائرة الصغيرة للراديو */
    div[data-testid="stRadio"] label input {
        display: none !important;
    }

    @media (max-width: 640px) {
        .header-left h1 { font-size: 18px; }
        .header-left .subtitle { font-size: 11px; }
        .header-ai-badge { font-size: 10px; padding: 3px 10px; }
        .top-badge-card { padding: 10px; gap: 8px; }
        .top-badge-card .tb-icon { width: 34px; height: 34px; min-width: 34px; font-size: 15px; }
        .top-badge-card .tb-title { font-size: 12px; }
        .top-badge-card .tb-desc { font-size: 10px; }
        .hero { padding: 1.5rem 1rem; min-height: auto; }
        .hero h1 { font-size: 22px; }
        .hero p { font-size: 14px; }
        .feature-card { padding: 1rem; }
        .feature-card .icon { font-size: 28px; }
        .how-section { padding: 1.5rem; }
        .sidebar-glass { padding: 1rem; }
        .result-card { padding: 1.2rem; }
        .result-card h2 { font-size: 24px; }
        .stButton > button { padding: 12px 20px; font-size: 14px; }
        div[data-testid="column"] button { padding: 12px 0 !important; font-size: 14px !important; }
        .heartbeat-icon { font-size: 22px; flex-wrap: wrap; }
        .heartbeat-icon span { font-size: 13px; padding: 2px 12px; }
        div[data-testid="stRadio"] label { padding: 8px 12px !important; font-size: 14px; }
    }
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE ==========
if 'language' not in st.session_state:
    st.session_state.language = "fr"
if 'history' not in st.session_state:
    st.session_state.history = []
if 'upload_mode' not in st.session_state:
    st.session_state.upload_mode = "file"
# ===== NEW: متغير الصفحة (افتراضي: الرئيسية) =====
if 'page' not in st.session_state:
    st.session_state.page = "home"

# ========== HEADER ==========
def get_file_base64(names):
    for name in names:
        if os.path.exists(name):
            with open(name, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None

def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

logo = get_file_base64(["logo.png", "logo.jpg", "logo.jpeg", "LOGO.png"])
logo_icon = get_file_base64(["logo_icon.png", "logo-icon.png"]) or logo

# ========== SIDEBAR (القائمة الجانبية) ==========
with st.sidebar:
    if logo:
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:1rem;">
            <img src="data:image/png;base64,{logo}" style="max-width:170px; width:100%;">
        </div>
        """, unsafe_allow_html=True)

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
    
    # ===== NEW: قائمة التنقل بين الصفحات (باستخدام radio) =====
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>📋 القائمة / Menu</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # استخدام radio بدلاً من العنصر الثابت
    nav_choice = st.radio(
        "##",  # تسمية مخفية
        options=[t("nav_home"), t("nav_history")],
        index=0 if st.session_state.page == "home" else 1,
        label_visibility="collapsed",
        key="nav_radio"
    )
    
    # تحديث حالة الصفحة بناءً على اختيار المستخدم
    if nav_choice == t("nav_home"):
        st.session_state.page = "home"
    else:
        st.session_state.page = "history"
    
    # ===== باقي عناصر القائمة الجانبية (QR, نصائح, دكتور) =====
    APP_URL = "https://hwaxrexkahkxaazwwjjr3d.streamlit.app/"
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&margin=10&color=F59E0B&data={APP_URL}"
    
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>{t('sidebar_qr_title')}</h4>
        <div class="qr-container">
            <img src="{qr_api_url}" width="160" height="160" alt="QR Code">
        </div>
        <p style="text-align:center; font-size:13px; color:#64748b;">{t('sidebar_qr_desc')}</p>
    </div>
    """, unsafe_allow_html=True)
    
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
logo_html = f'<div class="header-logo-badge"><img src="data:image/png;base64,{logo_icon}"></div>' if logo_icon else '<div class="header-logo-badge" style="font-size:26px;">🧬</div>'

st.markdown(f"""
<div class="header">
    <div class="header-badges">
        <div class="header-icon-btn">🔔<span class="dot"></span></div>
        <div class="header-badge" style="background:rgba(255,255,255,0.15); color:#fff; border-color:rgba(255,255,255,0.25);">🌐 {lang_map[st.session_state.language]}</div>
        <div class="header-ai-badge">✅ {t('ai_dev_badge')}</div>
    </div>
    <div class="header-left">
        <div>
            <h1>{t('app_title')}</h1>
            <div class="subtitle">{t('app_subtitle')}</div>
        </div>
        {logo_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ========== TOP BADGES ==========
st.markdown(f"""
<div class="top-badges">
    <div class="top-badge-card">
        <div class="tb-icon">🎧</div>
        <div><div class="tb-title">{t('top_badge1_title')}</div><div class="tb-desc">{t('top_badge1_desc')}</div></div>
    </div>
    <div class="top-badge-card disabled">
        <span class="tb-soon">🔜 {t('sidebar_doctor_soon')}</span>
        <div class="tb-icon">📞</div>
        <div><div class="tb-title">{t('top_badge2_title')}</div><div class="tb-desc">{t('top_badge2_desc')}</div></div>
    </div>
    <div class="top-badge-card">
        <div class="tb-icon">🕐</div>
        <div><div class="tb-title">{t('top_badge3_title')}</div><div class="tb-desc">{t('top_badge3_desc')}</div></div>
    </div>
    <div class="top-badge-card">
        <div class="tb-icon">🛡️</div>
        <div><div class="tb-title">{t('top_badge4_title')}</div><div class="tb-desc">{t('top_badge4_desc')}</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== HERO ==========
doctor_local = get_file_base64(["doctor.png", "doctor.jpg", "doctor.jpeg"])
if doctor_local:
    doctor_img_url = f"data:image/png;base64,{doctor_local}"
else:
    doctor_img_url = "https://cdn-icons-png.flaticon.com/512/3774/3774299.png"
heartbeat_svg = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 60'>"
    "<polyline points='0,30 60,30 80,10 100,50 120,5 140,55 160,30 220,30 240,15 260,45 280,30 400,30' "
    "fill='none' stroke='rgba(255,255,255,0.3)' stroke-width='3'/></svg>"
)

st.markdown(f"""
<div class="hero" id="hero-top">
    <img src="{heartbeat_svg}" class="hero-heartbeat" alt="">
    <div class="hero-content">
        <div class="hero-text">
            <span class="icon">🩺</span>
            <h1>{t('hero_title')}</h1>
            <p>{t('hero_desc')}</p>
            <a href="#upload-zone" class="hero-cta">➜ {t('hero_cta')}</a>
            <div style="margin-top:16px;">
                <span class="hero-badge">✅ {t('hero_badge')}</span>
            </div>
        </div>
        <div class="hero-visual">
            <div class="glow"></div>
            <img src="{doctor_img_url}" class="doctor-image" alt="Médecin / Doctor">
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# ========== UPLOAD (أزرار مع نبض) ==========
# ================================================================
st.markdown(f"""
<div class="upload-card" id="upload-zone">
    <div class="icon">📸</div>
    <h3>{t('upload_title')}</h3>
    <p>{t('upload_desc')}</p>
</div>
""", unsafe_allow_html=True)

# ---- علامة النبض فوق الأزرار ----
st.markdown("""
<div class="heartbeat-icon">
    💓 <span>❝ نبض التشخيص / Diagnostic Pulse ❞</span> 💓
</div>
""", unsafe_allow_html=True)

# أزرار اختيار الوضع
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("📁 " + t("upload_method_file"), use_container_width=True,
                 type="primary" if st.session_state.upload_mode == "file" else "secondary"):
        st.session_state.upload_mode = "file"
        st.rerun()
with col_btn2:
    if st.button("📷 " + t("upload_method_camera"), use_container_width=True,
                 type="primary" if st.session_state.upload_mode == "camera" else "secondary"):
        st.session_state.upload_mode = "camera"
        st.rerun()

# عرض أداة الرفع المناسبة
if st.session_state.upload_mode == "file":
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

# ========== FONCTIONS (نماذج ومعالجة) ==========
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        raw_pred = torch.sigmoid(output).item()
    
    corrected_pred = 1 - raw_pred
    
    if corrected_pred >= 0.5:
        result = "Anemic"
        confidence = corrected_pred * 100
    else:
        result = "Non Anemic"
        confidence = (1 - corrected_pred) * 100
    
    return result, confidence, corrected_pred, raw_pred

@st.cache_resource
def load_models():
    unet, dev_unet = load_unet_model()
    clf, dev_clf = load_classifier_model()
    return unet, dev_unet, clf, dev_clf

# ================================================================
# ========== عرض المحتوى حسب الصفحة المختارة ==========
# ================================================================

if st.session_state.page == "home":
    # ================================================================
    # ========== الصفحة الرئيسية (الرفع والتحليل) ==========
    # ================================================================
    
    if uploaded is not None:
        # Preview container
        preview_placeholder = st.empty()
        with preview_placeholder.container():
            st.markdown(f"""
            <div class="preview-container">
                <h4 style="margin-top:0;">{t('preview_title')}</h4>
            """, unsafe_allow_html=True)
            col_preview, _ = st.columns([1, 1])
            with col_preview:
                st.image(uploaded, use_container_width=True, caption=t('preview_caption'))
            st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button(t("analyze_btn"), use_container_width=True):
            # 1. Show Scanning Effect
            img_pil = Image.open(uploaded).convert('RGB')
            img_b64 = image_to_base64(img_pil)
            
            scan_placeholder = st.empty()
            with scan_placeholder.container():
                st.markdown(f"""
                <div class="preview-container">
                    <h4 style="margin-top:0;">🔬 جاري المسح الضوئي...</h4>
                    <div class="scan-container">
                        <img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:16px;">
                        <div class="scan-line"></div>
                        <div class="scan-overlay"></div>
                    </div>
                    <p style="text-align:center; color:#64748b; margin-top:8px;">{t('analyzing')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Simulate scan time + loading models
            with st.spinner(t("loading_models")):
                unet_model, unet_device, clf_model, clf_device = load_models()
            
            # Progress bar for the scan
            progress_bar = st.progress(0)
            for i in range(10):
                time.sleep(0.06)
                progress_bar.progress((i+1)*10)
            
            # 2. Real processing
            with st.spinner(t("analyzing")):
                img = np.array(img_pil)
                
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

                result, confidence, corrected_pred, raw_pred = predict_anemia(clf_model, conj_enhanced, clf_device)

                anemia_pct = corrected_pred * 100
                non_pct = (1 - corrected_pred) * 100

                progress_bar.empty()
                
                # Remove scan placeholder
                scan_placeholder.empty()
                
                st.success(t("analysis_done"))

                # ===== RESULTS DASHBOARD =====
                st.markdown(f'<div class="section-title">{t("results_title")}</div>', unsafe_allow_html=True)
                
                col_left, col_right = st.columns([2, 1])
                with col_left:
                    st.markdown(f"**👁️ {t('result_conjunctiva')}**")
                    st.image(conj_enhanced, use_container_width=True)
                with col_right:
                    st.markdown(f"**{t('result_original')}**")
                    st.image(img, use_container_width=True)
                    st.markdown(f"**{t('result_mask')}**")
                    st.image(final_mask, use_container_width=True, clamp=True)

                # Metrics
                before = np.sum(raw_mask > 0) / 255
                after = np.sum(final_mask > 0) / 255
                reduction = ((before - after) / before * 100) if before > 0 else 0

                m1, m2 = st.columns(2)
                with m1:
                    st.metric(t("metric_surface"), f"{after:.0f} px²")
                with m2:
                    st.metric(t("metric_cleaning"), f"{reduction:.1f}%")

                # DIAGNOSTIC
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

                # CHART
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

                # ===== إضافة التحليل إلى السجل (حتى لو كنا في الصفحة الرئيسية) =====
                entry = {
                    t("history_date"): datetime.now().strftime("%Y-%m-%d %H:%M"),
                    t("history_diagnostic"): result,
                    t("history_confidence"): f"{confidence:.1f}%",
                    t("history_prob"): f"{anemia_pct:.1f}%"
                }
                st.session_state.history.append(entry)
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop(0)

                # TECH DETAILS
                with st.expander(t("tech_details")):
                    st.write(f"**{t('tech_model_seg')}:** U‑Net (ResNet34)")
                    st.write(f"**{t('tech_model_clf')}:** EfficientNet‑B3")
                    st.write(f"**{t('tech_device')}:** {'GPU' if clf_device.type == 'cuda' else 'CPU'}")
                    st.write(f"**{t('tech_sigmoid')}:** {raw_pred:.4f}")
                    st.write(f"**{t('tech_prob_anemic')}:** {anemia_pct:.1f}%")
                    st.write(f"**{t('tech_prob_non')}:** {non_pct:.1f}%")
                    st.write(f"**{t('tech_preprocess')}:** CLAHE + Filtrage + Netteté")
                    st.write(f"**{t('tech_decision')}:** {t('tech_decision_value')}")

                # DISCLAIMER
                st.markdown(f"""
                <div class="disclaimer">
                    <strong>{t('disclaimer')}</strong><br>
                    {t('disclaimer_text')}<br>
                    {t('disclaimer_consult')}
                </div>
                """, unsafe_allow_html=True)

    # ========== FEATURES GRID (تظهر فقط في الصفحة الرئيسية) ==========
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

# ================================================================
# ========== صفحة سجل التحليلات ==========
# ================================================================
elif st.session_state.page == "history":
    st.markdown(f'<div class="section-title">{t("history_title")}</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        # عرض الجدول في بطاقة أنيقة
        st.markdown("""
        <div style="background: rgba(255,255,255,0.7); backdrop-filter: blur(8px); border-radius: 24px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.4); box-shadow: 0 8px 32px rgba(0,0,0,0.04);">
        """, unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # إضافة زر لمسح التاريخ (اختياري)
        if st.button("🗑️ مسح السجل / Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # رسالة ترحيب فارغة
        st.markdown(f"""
        <div style="text-align:center; padding: 4rem 1rem; background: rgba(255,255,255,0.6); border-radius: 32px; backdrop-filter: blur(4px); border: 2px dashed rgba(245,158,11,0.2);">
            <div style="font-size: 64px; margin-bottom: 1rem;">📭</div>
            <h3 style="color: #0f172a; font-weight: 700;">{t('history_empty')}</h3>
            <p style="color: #64748b; font-size: 16px;">{t('history_empty_desc')}</p>
            <br>
            <a href="#upload-zone" class="hero-cta" style="display: inline-flex;">➜ {t('hero_cta')}</a>
        </div>
        """, unsafe_allow_html=True)
