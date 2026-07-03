import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
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

# ========== الترجمة (ثلاث لغات) ==========
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

def t(key):
    lang = st.session_state.get("language", "fr")
    return LANGUAGES.get(lang, LANGUAGES["fr"]).get(key, key)

# ========== CSS (مدمج بالكامل من الصورة المطلوبة) ==========
# (تم حذف CSS للاختصار، لكنه موجود في الكود الكامل في الرد النهائي)
# ... (سيتم تضمين CSS كاملاً)

# ========== تهيئة Session State ==========
if 'language' not in st.session_state:
    st.session_state.language = "fr"
if 'history' not in st.session_state:
    st.session_state.history = []

# ========== تحميل النماذج ==========
@st.cache_resource
def load_models():
    unet, dev_unet = load_unet_model()
    clf, dev_clf = load_classifier_model()
    return unet, dev_unet, clf, dev_clf

# ========== دوال معالجة الصورة (بدون تحسينات تغير الجودة) ==========
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
            # لا نطبق أي تحسين إضافي على الصورة (CLAHE, blur, sharpening)
            return cropped, mask, (x, y, w, h)
    conj = cv2.bitwise_and(img, img, mask=mask)
    return conj, mask, None

# ========== دالة التصنيف مع التصحيح (عكس القيمة) ==========
def predict_anemia(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        raw = torch.sigmoid(out).item()

    # تصحيح النتيجة: النموذج يعطي قيماً معكوسة
    corrected = 1 - raw

    if corrected >= 0.5:
        return "Anemic", corrected * 100, raw, corrected
    else:
        return "Non Anemic", (1 - corrected) * 100, raw, corrected

# ========== الواجهة الجانبية ==========
with st.sidebar:
    lang_map = {"ar": "🇸🇦 العربية", "fr": "🇫🇷 Français", "en": "🇬🇧 English"}
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
    # ... (القائمة الجانبية ونصائح صحية وكود QR - كما في app (3).py)
    # (سيتم تضمينها في الكود النهائي)

# ========== المحتوى الرئيسي ==========
# ... (الهيدر، الهيرو، الإحصائيات، الميزات، شرح الخطوات، رفع الصورة)
# (سيتم تضمين كل ذلك مع الترجمة)

# ========== معالجة الصورة المرفوعة ==========
uploaded = None
# ... (جزء الرفع والكاميرا)

if uploaded is not None:
    # عرض معاينة الصورة
    # زر التحليل
    if st.button(t("analyze_btn")):
        with st.spinner(t("loading_models")):
            unet_model, unet_device, clf_model, clf_device = load_models()

        with st.spinner(t("analyzing")):
            img = np.array(Image.open(uploaded).convert('RGB'))
            img = cv2.flip(img, 1)  # انعكاس أفقي كما في النموذج

            # تجزئة U‑Net باستخدام Albumentations
            transform_unet = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            tensor = transform_unet(image=img)["image"].unsqueeze(0).to(unet_device)

            with torch.no_grad():
                raw_mask = torch.sigmoid(unet_model(tensor)).squeeze().cpu().numpy()
                raw_mask = (raw_mask > 0.5).astype(np.uint8) * 255
                raw_mask = cv2.resize(raw_mask, (img.shape[1], img.shape[0]))

            # استخراج الملتحمة بدون تحسينات
            conj, final_mask, bbox = extract_best_conjunctiva(img, raw_mask)

            # التصنيف مع التصحيح
            result, confidence, raw_pred, corrected_pred = predict_anemia(clf_model, conj, clf_device)

            anemia_pct = corrected_pred * 100
            non_pct = (1 - corrected_pred) * 100

            st.success(t("analysis_done"))

            # عرض النتائج (ثلاث أعمدة كما في الصورة)
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
                st.image(conj, use_container_width=True)

            # ... (باقي النتائج: مقاييس، تشخيص، مخطط، سجل، تفاصيل تقنية، تنويه)
            # (سيتم تضمينها كاملة)

# ========== تذييل الصفحة ==========
# ... (نفس التذييل من app (3).py)
