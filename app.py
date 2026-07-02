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
import time
from model_loader import load_unet_model, load_classifier_model

# ========== إعداد الصفحة ==========
st.set_page_config(
    page_title="AnemiCheck AI - كشف فقر الدم",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== ضبط اللغة الافتراضية للعربية (لتناسب الصورة) ==========
if 'language' not in st.session_state:
    st.session_state.language = "ar"  # العربية افتراضياً

# ========== الترجمة (تم تصغيرها قليلاً مع إضافة الكلمات الجديدة) ==========
LANGUAGES = {
    "ar": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "الذكاء الاصطناعي لصحة الدم",
        "badge_private": "أمين وموثوق",
        "badge_private_desc": "100% خصوصيتك محمية",
        "badge_available": "خدمة متاحة",
        "badge_available_desc": "24/7 نحن هنا من أجلك دائماً",
        "badge_call": "مكالمة مجانية",
        "badge_call_desc": "+213 123 456 789",
        "badge_help": "مساعدة فورية",
        "badge_help_desc": "تواصل مع فريق الدعم",
        "hero_title": "طبيبك معك بأقل من دقيقتين",
        "hero_desc": "تحليل ذكي لصورة العين لكشف الأنيميا بدقة وسرعة عالية",
        "hero_badge": "100% نتائج دقيقة وآمنة",
        "hero_cta": "ابدأ التحليل الآن",
        "how_title": "كيف يعمل التطبيق؟",
        "how_step1": "رفع الصورة",
        "how_step1_desc": "قم برفع صورة واضحة للعين (JPG, PNG)",
        "how_step2": "تحليل ذكي",
        "how_step2_desc": "يقوم الذكاء الاصطناعي بتحليل الصورة بدقة عالية",
        "how_step3": "عرض النتيجة",
        "how_step3_desc": "احصل على النتيجة فوراً مع تقرير مفصل",
        "feature_reliable": "موثوق ومعتمد",
        "feature_reliable_desc": "يساعد في الكشف المبكر عن الأنيميا",
        "feature_secure": "أمان",
        "feature_secure_desc": "بياناتك محمية بالكامل",
        "feature_trust": "ثقة عالية",
        "feature_trust_desc": "دقة تصل إلى 96%",
        "feature_fast": "سريع",
        "feature_fast_desc": "النتيجة في أقل من دقيقتين",
        "feature_smart": "ذكي",
        "feature_smart_desc": "تقنية ذكاء اصطناعي متقدمة",
        "upload_title": "رفع صورة العين",
        "upload_desc": "صورة واضحة للعين للكشف عن الأنيميا",
        "upload_method_file": "📁 رفع صورة",
        "upload_method_camera": "📷 تصوير",
        "analyze_btn": "🔍 تحليل الصورة",
        "loading_models": "⏳ جارٍ تحميل النماذج الذكية...",
        "analyzing": "⏳ جارٍ التحليل...",
        "analysis_done": "✅ تم التحليل بنجاح",
        "results_title": "📊 نتائج التحليل",
        "result_original": "🖼️ الصورة الأصلية",
        "result_mask": "🎭 منطقة العين المستخرجة",
        "result_cropped": "✂️ المنطقة المقطوعة (الجفن)",
        "diagnostic_title": "🩺 التشخيص",
        "diagnostic_anemic": "🩸 فقر دم",
        "diagnostic_anemic_desc": "يوجد فقر دم",
        "diagnostic_non_anemic": "✅ لا يوجد فقر دم",
        "diagnostic_non_anemic_desc": "لا يوجد فقر دم",
        "diagnostic_confidence": "📊 مستوى الثقة",
        "chart_title": "📈 توزيع الاحتمالات",
        "chart_non": "غير مصاب",
        "chart_anemic": "مصاب",
        "tech_details": "📘 التفاصيل التقنية",
        "tech_seg": "تقنية التجزئة المستخدمة",
        "tech_clf": "تقنية التصنيف المستخدمة",
        "tech_device": "الجهاز المستخدم",
        "tech_sigmoid": "قيمة السيجمويد الخام",
        "tech_prob_anemic": "احتمال فقر الدم",
        "tech_prob_non": "احتمال عدم الإصابة",
        "tech_decision": "القرار",
        "tech_decision_value": "مستقل تماماً، بدون تحيز",
        "disclaimer": "⚠️ تنويه طبي",
        "disclaimer_text": "هذه النتيجة صادرة عن نموذج ذكاء اصطناعي ولا تغني عن استشارة الطبيب المختص.",
        "disclaimer_consult": "استشر طبيباً للحصول على تشخيص دقيق.",
        "sidebar_health_title": "💡 نصائح صحية",
        "sidebar_health_1": "التغذية: ركز على الأطعمة الغنية بالحديد.",
        "sidebar_health_2": "فيتامين C: يحسن امتصاص الحديد.",
        "sidebar_health_3": "الترطيب: اشرب 1.5 لتر ماء يومياً.",
        "sidebar_health_4": "استشر الطبيب عند الشعور بالتعب الشديد.",
        "sidebar_version": "AnemiCheck v2.0 • الذكاء الاصطناعي الطبي",
        "lang_selector": "🌐 اللغة",
    },
    "fr": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "L'IA pour la santé sanguine",
        "badge_private": "Sécurisé et fiable",
        "badge_private_desc": "100% de confidentialité garantie",
        "badge_available": "Service disponible",
        "badge_available_desc": "24/7 nous sommes là pour vous",
        "badge_call": "Appel gratuit",
        "badge_call_desc": "+213 123 456 789",
        "badge_help": "Aide immédiate",
        "badge_help_desc": "Contactez notre équipe",
        "hero_title": "Votre médecin en moins de deux minutes",
        "hero_desc": "Analyse intelligente de l'image oculaire pour détecter l'anémie avec précision et rapidité",
        "hero_badge": "100% de résultats précis et sécurisés",
        "hero_cta": "Commencer l'analyse",
        "how_title": "Comment ça marche ?",
        "how_step1": "Téléchargement",
        "how_step1_desc": "Téléchargez une image claire de l'œil (JPG, PNG)",
        "how_step2": "Analyse intelligente",
        "how_step2_desc": "L'IA analyse l'image avec une haute précision",
        "how_step3": "Afficher le résultat",
        "how_step3_desc": "Obtenez le diagnostic instantanément",
        "feature_reliable": "Fiable et approuvé",
        "feature_reliable_desc": "Aide à la détection précoce de l'anémie",
        "feature_secure": "Sécurité",
        "feature_secure_desc": "Vos données sont protégées",
        "feature_trust": "Haute confiance",
        "feature_trust_desc": "Précision jusqu'à 96%",
        "feature_fast": "Rapide",
        "feature_fast_desc": "Résultat en moins de 2 minutes",
        "feature_smart": "Intelligent",
        "feature_smart_desc": "Technologie IA avancée",
        "upload_title": "Télécharger l'image",
        "upload_desc": "Image claire de l'œil pour la détection",
        "upload_method_file": "📁 Télécharger",
        "upload_method_camera": "📷 Appareil photo",
        "analyze_btn": "🔍 Analyser",
        "loading_models": "⏳ Chargement des modèles IA...",
        "analyzing": "⏳ Analyse en cours...",
        "analysis_done": "✅ Analyse terminée",
        "results_title": "📊 Résultats",
        "result_original": "🖼️ Image originale",
        "result_mask": "🎭 Zone extraite",
        "result_cropped": "✂️ Région coupée",
        "diagnostic_title": "🩺 Diagnostic",
        "diagnostic_anemic": "🩸 Anémie",
        "diagnostic_anemic_desc": "Anémie détectée",
        "diagnostic_non_anemic": "✅ Non Anémie",
        "diagnostic_non_anemic_desc": "Pas d'anémie détectée",
        "diagnostic_confidence": "📊 Niveau de confiance",
        "chart_title": "📈 Distribution",
        "chart_non": "Non Anémique",
        "chart_anemic": "Anémique",
        "tech_details": "📘 Détails techniques",
        "tech_seg": "Technique de segmentation",
        "tech_clf": "Technique de classification",
        "tech_device": "Appareil utilisé",
        "tech_sigmoid": "Valeur sigmoïde brute",
        "tech_prob_anemic": "Probabilité Anémie",
        "tech_prob_non": "Probabilité Non Anémie",
        "tech_decision": "Décision",
        "tech_decision_value": "100% autonome",
        "disclaimer": "⚠️ Avertissement médical",
        "disclaimer_text": "Ce résultat est généré par une IA et ne remplace pas un avis médical.",
        "disclaimer_consult": "Consultez un professionnel.",
        "sidebar_health_title": "💡 Conseils santé",
        "sidebar_health_1": "Alimentation : aliments riches en fer.",
        "sidebar_health_2": "Vitamine C : améliore l'absorption.",
        "sidebar_health_3": "Hydratation : 1,5 L d'eau par jour.",
        "sidebar_health_4": "Consultez en cas de fatigue.",
        "sidebar_version": "AnemiCheck v2.0 • IA médicale",
        "lang_selector": "🌐 Langue",
    },
    "en": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "AI for Blood Health",
        "badge_private": "Secure & Trusted",
        "badge_private_desc": "100% privacy protected",
        "badge_available": "Service Available",
        "badge_available_desc": "24/7 we are here for you",
        "badge_call": "Free Call",
        "badge_call_desc": "+213 123 456 789",
        "badge_help": "Instant Help",
        "badge_help_desc": "Chat with support",
        "hero_title": "Your doctor in less than two minutes",
        "hero_desc": "Intelligent analysis of eye images to detect anemia with high accuracy and speed",
        "hero_badge": "100% accurate and secure results",
        "hero_cta": "Start analysis now",
        "how_title": "How it works?",
        "how_step1": "Upload",
        "how_step1_desc": "Upload a clear eye image (JPG, PNG)",
        "how_step2": "Smart Analysis",
        "how_step2_desc": "AI analyzes the image with high precision",
        "how_step3": "View Result",
        "how_step3_desc": "Get the diagnosis instantly",
        "feature_reliable": "Reliable & Approved",
        "feature_reliable_desc": "Helps in early detection of anemia",
        "feature_secure": "Security",
        "feature_secure_desc": "Your data is fully protected",
        "feature_trust": "High Trust",
        "feature_trust_desc": "Accuracy up to 96%",
        "feature_fast": "Fast",
        "feature_fast_desc": "Result in less than 2 minutes",
        "feature_smart": "Smart",
        "feature_smart_desc": "Advanced AI technology",
        "upload_title": "Upload Eye Image",
        "upload_desc": "Clear eye image for anemia detection",
        "upload_method_file": "📁 Upload",
        "upload_method_camera": "📷 Camera",
        "analyze_btn": "🔍 Analyze",
        "loading_models": "⏳ Loading AI models...",
        "analyzing": "⏳ Analyzing...",
        "analysis_done": "✅ Analysis complete",
        "results_title": "📊 Results",
        "result_original": "🖼️ Original Image",
        "result_mask": "🎭 Extracted Area",
        "result_cropped": "✂️ Cropped Region",
        "diagnostic_title": "🩺 Diagnosis",
        "diagnostic_anemic": "🩸 Anemia",
        "diagnostic_anemic_desc": "Anemia detected",
        "diagnostic_non_anemic": "✅ No Anemia",
        "diagnostic_non_anemic_desc": "No anemia detected",
        "diagnostic_confidence": "📊 Confidence Level",
        "chart_title": "📈 Probability Distribution",
        "chart_non": "Non Anemic",
        "chart_anemic": "Anemic",
        "tech_details": "📘 Technical Details",
        "tech_seg": "Segmentation technique",
        "tech_clf": "Classification technique",
        "tech_device": "Device Used",
        "tech_sigmoid": "Raw Sigmoid Value",
        "tech_prob_anemic": "Anemia Probability",
        "tech_prob_non": "Non-Anemia Probability",
        "tech_decision": "Decision",
        "tech_decision_value": "100% autonomous",
        "disclaimer": "⚠️ Medical Disclaimer",
        "disclaimer_text": "This result is generated by an AI model and does not replace professional medical advice.",
        "disclaimer_consult": "Consult a healthcare professional.",
        "sidebar_health_title": "💡 Health Tips",
        "sidebar_health_1": "Diet: focus on iron-rich foods.",
        "sidebar_health_2": "Vitamin C: improves iron absorption.",
        "sidebar_health_3": "Hydration: drink 1.5 L of water daily.",
        "sidebar_health_4": "Consult a doctor if you experience fatigue.",
        "sidebar_version": "AnemiCheck v2.0 • Medical AI",
        "lang_selector": "🌐 Language",
    }
}

def t(key):
    lang = st.session_state.get("language", "ar")
    return LANGUAGES.get(lang, LANGUAGES["ar"]).get(key, key)

# ========== CSS والأنيميشن المميز ==========
st.markdown("""
<style>
    /* خطوط وأساسيات */
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800;900&display=swap');
    * { font-family: 'Tajawal', sans-serif; }

    /* خلفية متحركة مع دوائر ضبابية */
    .stApp {
        background: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 100%);
        position: relative;
        overflow-x: hidden;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -20%;
        width: 70%;
        height: 70%;
        background: radial-gradient(circle, rgba(220,38,38,0.08) 0%, transparent 70%);
        border-radius: 50%;
        animation: floatBlob 8s ease-in-out infinite alternate;
        z-index: 0;
        pointer-events: none;
    }
    .stApp::after {
        content: '';
        position: fixed;
        bottom: -30%;
        right: -10%;
        width: 60%;
        height: 60%;
        background: radial-gradient(circle, rgba(245,158,11,0.08) 0%, transparent 70%);
        border-radius: 50%;
        animation: floatBlob 10s ease-in-out infinite alternate-reverse;
        z-index: 0;
        pointer-events: none;
    }
    @keyframes floatBlob {
        0% { transform: translate(0, 0) scale(1); }
        100% { transform: translate(40px, 30px) scale(1.2); }
    }

    /* كل المحتوى فوق الخلفية */
    .main-content {
        position: relative;
        z-index: 1;
    }

    /* تأثير الكتابة (Typing) */
    .typing-effect {
        display: inline-block;
        overflow: hidden;
        white-space: nowrap;
        border-left: 3px solid #dc2626;
        animation: typing 3.5s steps(30, end), blink-caret 0.75s step-end infinite;
        font-weight: 800;
        color: #b91c1c;
        font-size: 28px;
    }
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    @keyframes blink-caret {
        from, to { border-color: transparent; }
        50% { border-color: #dc2626; }
    }

    /* البطاقات العلوية */
    .top-badges {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin: 20px 0 30px;
    }
    @media (max-width: 768px) { .top-badges { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 480px) { .top-badges { grid-template-columns: 1fr; } }
    .badge-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.5);
        box-shadow: 0 8px 32px rgba(0,0,0,0.04);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .badge-card:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(255,255,255,0.9);
        box-shadow: 0 16px 48px rgba(220,38,38,0.10);
        border-color: #dc2626;
    }
    .badge-card .icon {
        font-size: 36px;
        display: block;
        margin-bottom: 8px;
        animation: floatIcon 3s ease-in-out infinite;
    }
    @keyframes floatIcon {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    .badge-card .title { font-weight: 700; font-size: 18px; color: #0f172a; }
    .badge-card .desc { font-size: 14px; color: #64748b; }

    /* Hero */
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 50%, #2563eb 100%);
        border-radius: 32px;
        padding: 3rem 2.5rem;
        margin: 20px 0 30px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap-reverse;
        box-shadow: 0 20px 60px rgba(30,58,138,0.30);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
        transform: rotate(15deg);
    }
    .hero-text { flex: 1; min-width: 280px; z-index: 2; }
    .hero-text h1 { font-size: 38px; font-weight: 800; color: white; line-height: 1.3; }
    .hero-text h1 span { color: #FCD34D; }
    .hero-text p { color: #dbeafe; font-size: 16px; margin: 15px 0; max-width: 500px; }
    .hero-badge { background: rgba(16,185,129,0.2); color: #d1fae5; border: 1px solid rgba(52,211,153,0.3); padding: 8px 20px; border-radius: 40px; display: inline-block; font-weight: 600; }
    .hero-btn { background: linear-gradient(135deg, #F59E0B, #D97706); color: white; border: none; padding: 14px 40px; border-radius: 40px; font-weight: 700; font-size: 18px; box-shadow: 0 8px 24px rgba(245,158,11,0.4); transition: 0.3s; display: inline-block; text-decoration: none; margin-top: 10px; }
    .hero-btn:hover { transform: scale(1.05); box-shadow: 0 12px 32px rgba(245,158,11,0.6); }
    .hero-image { flex: 0 0 auto; z-index: 2; }
    .hero-image img { width: 200px; filter: drop-shadow(0 12px 28px rgba(0,0,0,0.2)); animation: floatIcon 4s ease-in-out infinite; border-radius: 50%; background: rgba(255,255,255,0.1); padding: 10px; }

    /* باقي العناصر */
    .section-title { font-size: 28px; font-weight: 800; color: #0f172a; margin: 30px 0 15px; border-right: 6px solid #dc2626; padding-right: 15px; }
    .how-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 25px; margin: 20px 0; }
    @media (max-width: 768px) { .how-steps { grid-template-columns: 1fr; } }
    .step-card { background: rgba(255,255,255,0.7); backdrop-filter: blur(8px); border-radius: 24px; padding: 25px; text-align: center; border: 1px solid rgba(255,255,255,0.4); transition: 0.3s; }
    .step-card:hover { transform: translateY(-5px); border-color: #dc2626; }
    .step-num { background: #dc2626; color: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: 800; margin: 0 auto 15px; }

    .features-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin: 20px 0; }
    @media (max-width: 768px) { .features-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 480px) { .features-grid { grid-template-columns: 1fr; } }
    .feature-item { background: white; border-radius: 18px; padding: 20px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.04); border: 1px solid #f1f5f9; transition: 0.3s; }
    .feature-item:hover { transform: translateY(-4px); box-shadow: 0 12px 24px rgba(0,0,0,0.06); border-color: #dc2626; }
    .feature-item .icon { font-size: 32px; }

    .upload-zone { background: rgba(255,255,255,0.8); backdrop-filter: blur(12px); border-radius: 30px; padding: 30px; border: 2px dashed #dc2626; text-align: center; margin: 20px 0; transition: 0.3s; }
    .upload-zone:hover { background: rgba(255,255,255,0.95); border-color: #b91c1c; box-shadow: 0 8px 30px rgba(220,38,38,0.08); }

    /* نتائج */
    .result-anemia { background: linear-gradient(135deg, #fff5f5, #fee2e2); border-radius: 24px; padding: 25px; border: 2px solid #dc2626; animation: popIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1); }
    .result-non-anemia { background: linear-gradient(135deg, #f0fff4, #dcfce7); border-radius: 24px; padding: 25px; border: 2px solid #16a34a; animation: popIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1); }
    @keyframes popIn { 0% { opacity: 0; transform: scale(0.9); } 100% { opacity: 1; transform: scale(1); } }

    .disclaimer-box { background: #fffbeb; border-radius: 16px; padding: 15px; border-right: 6px solid #f59e0b; margin-top: 30px; }

    /* إخفاء أي ذكر للنماذج */
    .model-card, .model-badge { display: none !important; }
    [data-testid="stMetric"] { background: rgba(255,255,255,0.7); backdrop-filter: blur(4px); border-radius: 16px; padding: 10px; }
    .stButton > button { background: linear-gradient(90deg, #dc2626, #ef4444); color: white; border: none; border-radius: 40px; padding: 12px; font-weight: 700; transition: 0.3s; }
    .stButton > button:hover { transform: scale(1.02); box-shadow: 0 8px 20px rgba(220,38,38,0.3); }
</style>
""", unsafe_allow_html=True)

# ========== الشعار ==========
def get_logo_base64():
    possible = ["logo.png", "logo.jpg", "logo.jpeg", "LOGO.png"]
    for p in possible:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None
logo_b64 = get_logo_base64()

# ========== الشريط الجانبي ==========
with st.sidebar:
    if logo_b64:
        st.markdown(f'<img src="data:image/png;base64,{logo_b64}" style="width:100%; max-width:200px; margin:0 auto; display:block;">', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="text-align:center; color:#dc2626;">🩸 AnemiCheck</h2>', unsafe_allow_html=True)
    
    lang_map = {"ar": "🇸🇦 العربية", "fr": "🇫🇷 Français", "en": "🇬🇧 English"}
    sel_lang = st.selectbox(t("lang_selector"), options=list(lang_map.keys()), format_func=lambda x: lang_map[x], index=list(lang_map.keys()).index(st.session_state.language))
    if sel_lang != st.session_state.language:
        st.session_state.language = sel_lang
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"### {t('sidebar_health_title')}")
    st.markdown(f"- {t('sidebar_health_1')}")
    st.markdown(f"- {t('sidebar_health_2')}")
    st.markdown(f"- {t('sidebar_health_3')}")
    st.markdown(f"- {t('sidebar_health_4')}")
    st.markdown("---")
    st.caption(t('sidebar_version'))

# ========== الواجهة الرئيسية ==========
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# 1. البطاقات العلوية
st.markdown(f"""
<div class="top-badges">
    <div class="badge-card"><span class="icon">🛡️</span><div class="title">{t('badge_private')}</div><div class="desc">{t('badge_private_desc')}</div></div>
    <div class="badge-card"><span class="icon">🕐</span><div class="title">{t('badge_available')}</div><div class="desc">{t('badge_available_desc')}</div></div>
    <div class="badge-card"><span class="icon">📞</span><div class="title">{t('badge_call')}</div><div class="desc">{t('badge_call_desc')}</div></div>
    <div class="badge-card"><span class="icon">💬</span><div class="title">{t('badge_help')}</div><div class="desc">{t('badge_help_desc')}</div></div>
</div>
""", unsafe_allow_html=True)

# 2. Hero
hero_img = "https://cdn-icons-png.flaticon.com/512/3774/3774299.png"
st.markdown(f"""
<div class="hero-container" id="hero-top">
    <div class="hero-text">
        <h1>{t('hero_title')}</h1>
        <p>{t('hero_desc')}</p>
        <span class="hero-badge">✅ {t('hero_badge')}</span><br><br>
        <a href="#upload-zone" class="hero-btn">🚀 {t('hero_cta')}</a>
    </div>
    <div class="hero-image">
        <img src="{hero_img}" alt="Doctor">
    </div>
</div>
""", unsafe_allow_html=True)

# 3. كيف يعمل
st.markdown(f'<div class="section-title">{t("how_title")}</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="how-steps">
    <div class="step-card"><div class="step-num">1</div><h4>{t('how_step1')}</h4><p>{t('how_step1_desc')}</p></div>
    <div class="step-card"><div class="step-num">2</div><h4>{t('how_step2')}</h4><p>{t('how_step2_desc')}</p></div>
    <div class="step-card"><div class="step-num">3</div><h4>{t('how_step3')}</h4><p>{t('how_step3_desc')}</p></div>
</div>
""", unsafe_allow_html=True)

# 4. مميزات إضافية
st.markdown(f"""
<div class="features-grid">
    <div class="feature-item"><span class="icon">✅</span><h4>{t('feature_reliable')}</h4><p>{t('feature_reliable_desc')}</p></div>
    <div class="feature-item"><span class="icon">🔒</span><h4>{t('feature_secure')}</h4><p>{t('feature_secure_desc')}</p></div>
    <div class="feature-item"><span class="icon">⭐</span><h4>{t('feature_trust')}</h4><p>{t('feature_trust_desc')}</p></div>
    <div class="feature-item"><span class="icon">⚡</span><h4>{t('feature_fast')}</h4><p>{t('feature_fast_desc')}</p></div>
    <div class="feature-item"><span class="icon">🧠</span><h4>{t('feature_smart')}</h4><p>{t('feature_smart_desc')}</p></div>
</div>
""", unsafe_allow_html=True)

# 5. منطقة الرفع
st.markdown(f"""
<div class="upload-zone" id="upload-zone">
    <span style="font-size:48px;">👁️📤</span>
    <h3 style="color:#b91c1c;">{t('upload_title')}</h3>
    <p style="color:#64748b;">{t('upload_desc')}</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    option = st.radio("", [t("upload_method_file"), t("upload_method_camera")], horizontal=True, label_visibility="collapsed")
with col2:
    if option == t("upload_method_file"):
        uploaded = st.file_uploader("", type=["jpg","png","jpeg"], label_visibility="collapsed", key="file_uploader")
    else:
        uploaded = st.camera_input("", label_visibility="collapsed", key="camera_input")

# =============================================================
# ===== دوال المعالجة (ممنوع اللمس - محفوظة كما هي) =====
# =============================================================
def extract_eyelid_region(img, mask, padding=15):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        x = max(0, x - padding); y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        if w > 0 and h > 0:
            return img[y:y+h, x:x+w], mask[y:y+h, x:x+w], (x, y, w, h)
    return img, mask, None

def predict_anemia(model, image, device, use_inversion=True, threshold=0.5):
    """
    تصنيف الصورة باستخدام النموذج.
    - use_inversion: إذا كان True، نعكس النتيجة (1 - raw_pred) لأن النموذج معكوس.
    - threshold: العتبة للفصل بين الفئتين (0.5 افتراضياً).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        raw_pred = torch.sigmoid(model(img_tensor)).item()
    
    # تطبيق الانعكاس إذا طُلب
    if use_inversion:
        corrected_pred = 1 - raw_pred
    else:
        corrected_pred = raw_pred
    
    # القرار بناءً على العتبة
    if corrected_pred >= threshold:
        result = "Anemic"
        confidence = corrected_pred * 100  # نعبر عن الثقة كنسبة مئوية
    else:
        result = "Non Anemic"
        confidence = (1 - corrected_pred) * 100
    
    return result, confidence, corrected_pred, raw_pred

# =============================================================
# ===== معالجة الصورة المرفوعة =====
# =============================================================
if uploaded is not None:
    with st.spinner(t("loading_models")):
        unet_model, unet_device = load_unet_model()
        clf_model, clf_device = load_classifier_model()

    with st.spinner(t("analyzing")):
        img = np.array(Image.open(uploaded).convert('RGB'))
        
        # تجزئة
        transform_unet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])
        img_tensor = transform_unet(img).unsqueeze(0).to(unet_device)
        with torch.no_grad():
            raw_mask = torch.sigmoid(unet_model(img_tensor)).squeeze().cpu().numpy()
            raw_mask = (raw_mask > 0.5).astype(np.uint8) * 255
            raw_mask = cv2.resize(raw_mask, (img.shape[1], img.shape[0]))
        
        # استخراج بالقص
        cropped_img, cropped_mask, bbox = extract_eyelid_region(img, raw_mask, padding=15)
        
        # تصنيف
        result, confidence, corrected_pred, raw_pred = predict_anemia(clf_model, cropped_img, clf_device)
        anemia_pct = corrected_pred * 100
        non_pct = (1 - corrected_pred) * 100

        st.success(t("analysis_done"))

        # عرض النتائج
        st.markdown(f'<div class="section-title">{t("results_title")}</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(f"**{t('result_original')}**"); st.image(img, use_container_width=True)
        with col2: st.markdown(f"**{t('result_mask')}**"); st.image(raw_mask, use_container_width=True, clamp=True)
        with col3: st.markdown(f"**{t('result_cropped')}**"); st.image(cropped_img, use_container_width=True)

        # تشخيص
        st.markdown(f'<div class="section-title">{t("diagnostic_title")}</div>', unsafe_allow_html=True)
        col_r, col_c = st.columns(2)
        with col_r:
            if result == "Anemic":
                st.markdown(f'<div class="result-anemia"><h2 style="color:#dc2626;">{t("diagnostic_anemic")}</h2><p>{t("diagnostic_anemic_desc")}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-non-anemia"><h2 style="color:#16a34a;">{t("diagnostic_non_anemic")}</h2><p>{t("diagnostic_non_anemic_desc")}</p></div>', unsafe_allow_html=True)
        with col_c:
            st.metric(t("diagnostic_confidence"), f"{confidence:.1f}%")
            st.progress(int(confidence))

        # مخطط
        st.markdown(f'<div class="section-title">{t("chart_title")}</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar([t('chart_non'), t('chart_anemic')], [non_pct, anemia_pct], color=['#10b981','#dc2626'], width=0.5)
        ax.set_ylim(0,100); ax.set_ylabel('نسبة %'); ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, [non_pct, anemia_pct]):
            ax.text(bar.get_x()+bar.get_width()/2, v+2, f'{v:.1f}%', ha='center', fontweight='bold')
        st.pyplot(fig)

        # تفاصيل تقنية (بدون أسماء نماذج)
        with st.expander(t("tech_details")):
            st.write(f"**{t('tech_seg')}:** تجزئة بالذكاء الاصطناعي (Deep Learning)")
            st.write(f"**{t('tech_clf')}:** تصنيف بالشبكات العصبية العميقة")
            st.write(f"**{t('tech_device')}:** {'GPU' if clf_device.type == 'cuda' else 'CPU'}")
            st.write(f"**{t('tech_sigmoid')}:** {raw_pred:.4f}")
            st.write(f"**{t('tech_prob_anemic')}:** {anemia_pct:.1f}%")
            st.write(f"**{t('tech_prob_non')}:** {non_pct:.1f}%")
            st.write(f"**{t('tech_decision')}:** {t('tech_decision_value')}")

        # تنويه
        st.markdown(f"""
        <div class="disclaimer-box">
            <strong>{t('disclaimer')}</strong><br>
            {t('disclaimer_text')}<br>
            {t('disclaimer_consult')}
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
